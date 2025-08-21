import numpy as np
import random
from output_layer_generator import NeuralNetworkGenerator

def generate_layer_widths(input_size, num_layers, output_size=None, min_width=100, max_width=500, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if output_size is None:
        output_size = min_width
    if num_layers == 1:
        return [output_size]
    hidden_widths = np.random.randint(
        max(min_width, input_size // 10),
        max(max_width, input_size // 2),
        num_layers - 1
    )
    widths = list(hidden_widths) + [output_size]
    return widths

def simple_continuous_network_matrix(N, T, input_vector, L, widths, random_seed=None):
    """
    Simple neural network that outputs N×T matrix of continuous values
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    current = input_vector.reshape(-1, 1)
    if L == 0:
        input_dim = current.shape[0]
        output_dim = N * T
        std = np.sqrt(2.0 / input_dim)
        W = np.random.normal(0, std, (output_dim, input_dim))
        b = np.random.normal(0, 0.01, (output_dim, 1))
        z = W @ current + b
        return z.reshape(N, T)
    for layer in range(L):
        input_dim = current.shape[0]
        if layer == L - 1:
            output_dim = N * T
        else:
            output_dim = widths[layer]
        std = np.sqrt(2.0 / input_dim)
        W = np.random.normal(0, std, (output_dim, input_dim))
        b = np.random.normal(0, 0.01, (output_dim, 1))
        z = W @ current + b
        if layer < L - 1:
            current = np.maximum(0, z)
        else:
            current = z
    return current.reshape(N, T)

class LatentSpaceGenerator:
    def __init__(self, N, T, d, random_seed=None):
        self.N = N
        self.T = T
        self.d = d
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    def random_covariance_matrix(self, n):
        A = np.random.randn(n, n)
        return A @ A.T
    def generate_latent_space(self):
        mean_latent_row = [random.random() for _ in range(self.d)]
        mean_latent_col = [random.random() for _ in range(self.d)]
        cov_latent_row = self.random_covariance_matrix(self.d)
        cov_latent_col = self.random_covariance_matrix(self.d)
        U = np.random.multivariate_normal(mean_latent_row, cov_latent_row, size=self.N)
        V = np.random.multivariate_normal(mean_latent_col, cov_latent_col, size=self.T)
        return U, V

class DataMatrixGenerator:
    def __init__(self, N, T, random_seed=None):
        self.N = N
        self.T = T
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    def generate_data_matrix(self, U, V, L, widths=None):
        input_vector = np.concatenate([U.flatten(), V.flatten()])
        input_size = input_vector.shape[0]
        output_size = self.N * self.T
        if widths is None:
            widths = generate_layer_widths(input_size, L, output_size=output_size, random_seed=self.random_seed)
        # Use the neural network for both L==0 and L>0
        return simple_continuous_network_matrix(self.N, self.T, input_vector, L, widths, random_seed=self.random_seed)

class MaskingMatrixGenerator:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    def generate_probability_matrix(self, X, U, V, L, widths=None, seed_base=42):
        N, T = X.shape
        X_flat = X.flatten()
        U_flat = U.flatten()
        V_flat = V.flatten()
        input_vector = np.concatenate([X_flat, U_flat, V_flat])
        input_size = input_vector.shape[0]
        output_size = 2  # For binary mask
        if widths is None:
            widths = generate_layer_widths(input_size, L, output_size=output_size, random_seed=self.random_seed)
        if L == 0:
            np.random.seed(seed_base)
            W = np.random.normal(0, 1.0 / np.sqrt(input_size), (N * T, input_size))
            b = np.random.normal(0, 0.01, (N * T, 1))
            z = W @ input_vector.reshape(-1, 1) + b
            prob = 1 / (1 + np.exp(-z))
            return prob.reshape(N, T)
        else:
            prob_matrix = np.zeros((N, T))
            for i in range(N):
                for t in range(T):
                    unique_seed = seed_base + i * T + t
                    nn_ij = NeuralNetworkGenerator(
                        input_size_p=input_size,
                        layer_sizes_L=L,
                        widths_W=widths,
                        activation='relu',
                        random_seed=unique_seed
                    )
                    probs = nn_ij.generate_output(input_vector, verbose=False)
                    prob_matrix[i, t] = probs[0]
            return prob_matrix
    def generate_binary_matrices(self, prob_matrix, num_masks=1, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        N, T = prob_matrix.shape
        masks = []
        for _ in range(num_masks):
            random_values = np.random.random(prob_matrix.shape)
            binary_matrix = (random_values < prob_matrix).astype(int)
            masks.append(binary_matrix)
        return masks

def run_complete_pipeline(
    N=50, T=50, d=5, 
    L_X=0, 
    L_mask=3, 
    latent_seed=42, data_seed=42, mask_seed=42,
    num_masks=1
):
    # Step 1: Latent space
    latent_gen = LatentSpaceGenerator(N, T, d, random_seed=latent_seed)
    U, V = latent_gen.generate_latent_space()
    print(f"Step 1: U shape {U.shape}, V shape {V.shape}")
    # Step 2: Data matrix
    data_gen = DataMatrixGenerator(N, T, random_seed=data_seed)
    # Generate widths for X
    input_vector_X = np.concatenate([U.flatten(), V.flatten()])
    input_size_X = input_vector_X.shape[0]
    output_size_X = N * T
    widths_X = generate_layer_widths(input_size_X, L_X, output_size=output_size_X, random_seed=data_seed)
    X = data_gen.generate_data_matrix(U, V, L_X, widths=widths_X)
    print(f"Step 2: X shape {X.shape}")
    # Step 3: Probability and masking matrices
    mask_gen = MaskingMatrixGenerator(random_seed=mask_seed)
    # Generate widths for mask
    X_flat = X.flatten()
    U_flat = U.flatten()
    V_flat = V.flatten()
    input_vector_mask = np.concatenate([X_flat, U_flat, V_flat])
    input_size_mask = input_vector_mask.shape[0]
    output_size_mask = 2
    widths_mask = generate_layer_widths(input_size_mask, L_mask, output_size=output_size_mask, random_seed=mask_seed)
    prob_matrix = mask_gen.generate_probability_matrix(X, U, V, L_mask, widths=widths_mask)
    print(f"Step 3: Probability matrix shape {prob_matrix.shape}")
    masks = mask_gen.generate_binary_matrices(prob_matrix, num_masks=num_masks, random_seed=mask_seed)
    print(f"Generated {num_masks} masking matrices, each shape: {masks[0].shape}")
    return U, V, X, masks, prob_matrix
