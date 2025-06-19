import pandas as pd
import numpy as np
import networkx as nx
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import beta
import warnings

# --- Helper functions for sampling ---
def sample_log_uniform(low, high, size=1, base=np.e):
    """Sample from a log-uniform distribution between low and high."""
    return np.power(base, np.random.uniform(np.log(low)/np.log(base), 
                                            np.log(high)/np.log(base), 
                                            size))

def sample_discretized_log_normal(mean, min_val, size=1):
    """Sample from a discretized log-normal distribution with mean and minimum value."""
    sigma = 0.8
    val = np.random.lognormal(mean=np.log(mean), sigma=sigma, size=size)
    return np.maximum(min_val, np.round(val)).astype(int)

def sample_exponential(scale, min_val, size=1):
    """Sample from an exponential distribution with scale and minimum value."""
    return min_val + np.round(np.random.exponential(scale=scale, size=size)).astype(int)

# --- Graph Generation Functions ---
def generate_mlp_dropout_dag(num_nodes: int, num_layers: int, edge_dropout_prob: float = 0.2) -> nx.DiGraph:
    G = nx.DiGraph(); nodes_per_layer_base = np.zeros(num_layers, dtype=int) + 1
    nodes_remaining = num_nodes - num_layers
    if nodes_remaining > 0:
        splits = np.sort(np.random.choice(nodes_remaining + num_layers - 1, num_layers - 1, replace=False))
        layer_sizes = np.diff(np.concatenate(([0], splits, [nodes_remaining + num_layers - 1]))) - 1
        nodes_per_layer_base += layer_sizes
    node_indices = np.arange(num_nodes)
    nodes_per_layer = np.split(node_indices, np.cumsum(nodes_per_layer_base)[:-1])
    for i, layer_nodes in enumerate(nodes_per_layer):
        for node in layer_nodes: G.add_node(node, layer=i)
    for i in range(num_layers - 1):
        for u in nodes_per_layer[i]:
            for v in nodes_per_layer[i+1]:
                if np.random.rand() > edge_dropout_prob: G.add_edge(u, v)
    return G

def generate_scale_free_dag(num_nodes: int, m_edges: int = 2) -> nx.DiGraph:
    undirected_G = nx.barabasi_albert_graph(n=num_nodes, m=m_edges)
    G = nx.DiGraph(); G.add_nodes_from(undirected_G.nodes())
    for u, v in undirected_G.edges():
        if u < v: G.add_edge(u, v)
        else: G.add_edge(v, u)
    for node in G.nodes(): G.nodes[node]['layer'] = node % 5
    return G

def sample_dag_structure(config: dict) -> nx.DiGraph:
    graph_type = np.random.choice(config['graph_generation_method'])
    num_nodes = int(sample_log_uniform(config['num_nodes_low'], config['num_nodes_high'])[0])
    if graph_type == 'MLP-Dropout':
        num_layers = sample_discretized_log_normal(mean=config['mlp_num_layers_mean'], min_val=2)[0]
        return generate_mlp_dropout_dag(num_nodes=num_nodes, num_layers=num_layers)
    elif graph_type == 'Scale-Free':
        m_edges = np.random.randint(1, 5)
        return generate_scale_free_dag(num_nodes=num_nodes, m_edges=m_edges)

# --- Functional Mechanism Functions ---
def softplus(x): return np.maximum(0, x) + np.log(1 + np.exp(-np.abs(x)))
SELU_LAMBDA, SELU_ALPHA = 1.0507, 1.67326
ACTIVATION_FUNCTIONS = {'identity': lambda x: x, 'tanh': np.tanh, 'leaky_relu': lambda x: np.maximum(0.01 * x, x),'elu': lambda x: np.where(x > 0, x, 0.5 * (np.exp(x) - 1)), 'silu': lambda x: x / (1 + np.exp(-x)),'sine': np.sin, 'relu': lambda x: np.maximum(0, x), 'relu6': lambda x: np.minimum(np.maximum(0, x), 6),'selu': lambda x: SELU_LAMBDA * np.where(x > 0, x, SELU_ALPHA * (np.exp(x) - 1)),'softplus': softplus, 'hardtanh': lambda x: np.maximum(-1, np.minimum(1, x)), 'sign': np.sign,'rbf': lambda x: np.exp(-np.square(x)), 'exp': lambda x: np.exp(np.clip(x, -10, 10)),'sqrt_abs': lambda x: np.sqrt(np.abs(x)), 'indicator_abs_le_1': lambda x: np.where(np.abs(x) <= 1, 1, 0),'square': np.square, 'absolute': np.abs,}
class RandomFourierFeatureFunction:
    def __init__(self, N=256):
        u = np.random.uniform(0.7, 3.0); self.N = N
        self.b = np.random.uniform(0, 2 * np.pi, N); self.a = np.random.uniform(0, N, N)
        self.w = self.a ** (-np.exp(u))
    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = np.random.randn(self.N)
        if x.ndim == 1: x = x[:, np.newaxis]
        phi_x = (self.w / np.linalg.norm(self.w)) * np.sin(self.a * x + self.b)
        return np.dot(phi_x, z)

def assign_functional_mechanisms(dag: nx.DiGraph, config: dict):
    function_choices = config['scm_activation_functions'] + ['random_fourier']
    for node in dag.nodes():
        choice = np.random.choice(function_choices)
        if choice != 'random_fourier' and np.random.rand() < config['function_type_mixture_ratio']:
            func_name = np.random.choice(config['scm_activation_functions']); parents = list(dag.predecessors(node))
            dag.nodes[node].update({'type': 'scm', 'function_name': func_name, 'function': ACTIVATION_FUNCTIONS[func_name], 'weights': np.random.randn(len(parents)), 'bias': np.random.randn()})
        elif choice == 'random_fourier':
            parents = list(dag.predecessors(node))
            dag.nodes[node].update({'type': 'random_fourier', 'function_name': 'RandomFourierFunc', 'function': RandomFourierFeatureFunction(), 'parent_selector': np.random.randint(len(parents)) if parents else None})
        else:
            params = {'n_estimators': sample_exponential(config['xgb_n_estimators_exp_scale'], 1)[0], 'max_depth': sample_exponential(config['xgb_max_depth_exp_scale'], 2)[0], 'n_jobs': 1}
            model = xgb.XGBRegressor(**params)
            dag.nodes[node].update({'type': 'tree', 'function_name': 'XGBoost', 'function': model, 'params': params})

# --- Matrix Generation Functions ---
def generate_complete_matrix(config: dict) -> pd.DataFrame:
    """Generates a single, complete n x m matrix using the full SCM process."""
    dag = sample_dag_structure(config)
    assign_functional_mechanisms(dag, config)
    n_rows = int(sample_log_uniform(config['num_rows_low'], config['num_rows_high'])[0])
    sorted_nodes = list(nx.topological_sort(dag))
    node_data = {node: np.zeros(n_rows) for node in sorted_nodes}
    
    for node in sorted_nodes:
        parents = list(dag.predecessors(node))
        node_info = dag.nodes[node]
        if not parents:
            noise_type = np.random.choice(config['root_node_noise_dist'])
            if noise_type == 'Normal':
                node_data[node] = np.random.randn(n_rows)
            elif noise_type == 'Uniform':
                node_data[node] = np.random.uniform(-1, 1, n_rows)
        else:
            parent_values = np.vstack([node_data[p] for p in parents]).T
            if node_info['type'] == 'scm':
                linear_combination = np.dot(parent_values, node_info['weights']) + node_info['bias']
                node_data[node] = node_info['function'](linear_combination)
            elif node_info['type'] == 'random_fourier':
                selected_parent_data = parent_values[:, node_info['parent_selector']]
                node_data[node] = node_info['function'](selected_parent_data)
            elif node_info['type'] == 'tree':
                fake_targets = np.random.randn(n_rows)
                node_info['function'].fit(parent_values, fake_targets)
                node_data[node] = node_info['function'].predict(parent_values)
                
    num_scm_nodes = dag.number_of_nodes()
    m_cols = np.random.randint(config['num_cols_low'], min(num_scm_nodes + 1, config['num_cols_high']))
    final_cols = np.random.choice(list(node_data.keys()), m_cols, replace=False)
    matrix = pd.DataFrame({f"feature_{col}": node_data[col] for col in final_cols})

    if np.random.rand() < config['apply_feature_warping_prob']:
        print("Feature Warping with Beta Distribution")
        scaler = MinMaxScaler()
        for col in matrix.columns:
            # Scale data to [0, 1] range required by Beta 
            col_data = scaler.fit_transform(matrix[[col]])
            
            # Sample shape parameters (alpha and beta) for the Beta 
            a, b = np.random.rand() * 5 + 0.5, np.random.rand() * 5 + 0.5
            
            # Clip to avoid issues with 0 and 1
            epsilon = 1e-10
            clipped_data = np.clip(col_data, epsilon, 1 - epsilon)
            warped_data = beta.ppf(clipped_data, a, b)
            
            # Scale back to original range
            matrix[col] = scaler.inverse_transform(warped_data.reshape(-1, 1)).flatten()
            
    if np.random.rand() < config['apply_quantization_prob']:
        col_to_quantize = np.random.choice(matrix.columns)
        num_bins = np.random.randint(2, 20)
        try:
            matrix[col_to_quantize] = pd.qcut(matrix[col_to_quantize], q=num_bins, labels=False, duplicates='drop')
        except ValueError:
            pass # Ignore if quantization fails
            
    return matrix

# --- Missingness Functions ---
def _logit(x):
    return 1 / (1 + np.exp(-x))

def induce_mcar(matrix: pd.DataFrame, p: float):
    """
    Induces missing completely at random (MCAR) by randomly masking a specified percentage of cells.
    """
    mat = matrix.copy(); total_cells = mat.size; num_missing = int(total_cells * p)
    indices = np.random.choice(total_cells, num_missing, replace=False)
    row_indices, col_indices = np.unravel_index(indices, mat.shape)
    mat.values[row_indices, col_indices] = np.nan
    return mat

def induce_mar(matrix: pd.DataFrame, p: float):
    """
    Induces missing at random (MAR) by masking values based on a linear combination of predictor columns and a random choice of beta_0.
    """
    mat = matrix.copy(); num_cols = mat.shape[1]
    num_predictors = np.random.randint(1, max(2, num_cols // 2))
    predictor_cols = mat.sample(n=num_predictors, axis=1).columns
    target_cols = mat.columns.difference(predictor_cols)
    betas = np.random.randn(len(predictor_cols))
    log_odds = np.dot(mat[predictor_cols].fillna(0), betas)
    beta_0 = -np.quantile(log_odds, 1 - p)
    probs = _logit(log_odds + beta_0)
    for col in target_cols:
        mask = np.random.rand(len(mat)) < probs
        mat.loc[mask, col] = np.nan
    return mat

def induce_mnar(matrix: pd.DataFrame, p: float):
    """
    Induces missing not at random (MNAR) by selectively masking values based on the target column and a random choice of beta_1.
    """
    mat = matrix.copy()
    target_cols = mat.sample(n=np.random.randint(1, mat.shape[1] + 1), axis=1).columns
    for col in target_cols:
        beta_1 = np.random.choice([-2, -1, 1, 2])
        log_odds = mat[col] * beta_1
        beta_0 = -np.quantile(log_odds, 1 - p) if not log_odds.isnull().all() else 0
        probs = _logit(log_odds + beta_0)
        mask = np.random.rand(len(mat)) < probs
        mat.loc[mask, col] = np.nan
    return mat

# --- Transformation Functions ---
def create_cell_as_sample_dataset(matrix: pd.DataFrame):
    """
    Transforms incomplete matrix into supervised dataset. Each cell becomeess a data point with its features engineered from its
    position and the statistical context of its row+ column and the global matrix.
    """
    n_rows, m_cols = matrix.shape; samples = []
    row_stats = {'mean': matrix.mean(axis=1), 'std': matrix.std(axis=1), 'min': matrix.min(axis=1), 'max': matrix.max(axis=1), 'count': matrix.count(axis=1)}
    col_stats = {'mean': matrix.mean(axis=0), 'std': matrix.std(axis=0), 'min': matrix.min(axis=0), 'max': matrix.max(axis=0), 'count': matrix.count(axis=0)}
    observed_values = matrix.values[~np.isnan(matrix.values)]; global_mean = observed_values.mean(); global_std = observed_values.std(); global_count = len(observed_values)
    for i in range(n_rows):
        for j in range(m_cols):
            feature_vector = {'row_idx': i, 'col_idx': j, 'row_mean': row_stats['mean'].iloc[i], 'row_std': row_stats['std'].iloc[i], 'row_min': row_stats['min'].iloc[i], 'row_max': row_stats['max'].iloc[i], 'row_count_observed': row_stats['count'].iloc[i], 'col_mean': col_stats['mean'].iloc[j], 'col_std': col_stats['std'].iloc[j], 'col_min': col_stats['min'].iloc[j], 'col_max': col_stats['max'].iloc[j], 'col_count_observed': col_stats['count'].iloc[j], 'global_mean': global_mean, 'global_std': global_std, 'global_count_observed': global_count, 'target_value': matrix.iloc[i, j]}
            samples.append(feature_vector)
    return pd.DataFrame(samples)

def discretize_target(df: pd.DataFrame, target_col: str, num_bins: int):
    df_copy = df.copy(); binned_col_name = f"{target_col}_binned"
    df_copy[binned_col_name] = pd.qcut(df_copy[target_col], q=num_bins, labels=False, duplicates='drop')
    return df_copy