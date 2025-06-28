import numpy as np
import random
from modules import DataMatrixGenerator, MaskingMatrixGenerator
from scipy.linalg import svd

### Hierarchical generation (TILING) pipeline ###

def compute_first_left_right_eigenvectors(matrix):
    # SVD: matrix = U S V^T
    U, S, Vh = svd(matrix, full_matrices=False)
    left = U[:, 0]  # First left singular vector
    right = Vh[0, :]  # First right singular vector
    return left, right

def run_hierarchical_generation(
    data_matrices,  # shape: (J, K, N, T)
    masking_matrices,  # shape: (J, K, N, T)
    B,  # number of pairs to sample
    data_L=1,  # number of layers for data matrix generation
    mask_L=1,  # number of layers for mask generation
    num_big_masks=1,  # number of big mask matrices to generate
    data_seed=None,
    mask_seed=None,
    pair_seed=None
):
    """
    Hierarchical generation as described in the plan.
    Returns:
        - big_data_matrix: (N*J, T*K)
        - big_mask_matrices: list of (N*J, T*K) binary matrices
        - P_M: (J, K) probability mask
        - sampled_pairs: list of (j, k) tuples
    """
    J, K, N, T = data_matrices.shape
    # 1. Compute eigenvectors for all data and masking matrices
    data_eigvecs = [[compute_first_left_right_eigenvectors(data_matrices[j, k]) for k in range(K)] for j in range(J)]
    mask_eigvecs = [[compute_first_left_right_eigenvectors(masking_matrices[j, k]) for k in range(K)] for j in range(J)]

    # 2. Randomly sample B pairs
    all_pairs = [(j, k) for j in range(J) for k in range(K)]
    if pair_seed is not None:
        random.seed(pair_seed)
    sampled_pairs = random.sample(all_pairs, B)

    # 3. Construct v* and u*
    v_star_list = []
    u_star_list = []
    for (j, k) in sampled_pairs:
        # Data matrix eigenvectors
        data_left, data_right = data_eigvecs[j][k]
        v_star_list.append(data_left)
        v_star_list.append(data_right)
        # Masking matrix eigenvectors
        mask_left, mask_right = mask_eigvecs[j][k]
        u_star_list.append(mask_left)
        u_star_list.append(mask_right)
    v_star = np.concatenate(v_star_list)  # shape: B*(N+T)
    u_star = np.concatenate(u_star_list)  # shape: B*(N+T)

    # 4. Data matrix generation
    data_gen = DataMatrixGenerator(N*J, T*K, random_seed=data_seed)
    # widths for data matrix
    widths_data = None  # Let generator decide
    big_data_matrix = data_gen.generate_data_matrix(
        U=v_star.reshape(-1, 1),  # as column vector
        V=np.zeros((1,)),  # not used, but required by signature
        L=data_L,
        widths=widths_data
    ).reshape(N*J, T*K)

    # 5. Masking matrix generation
    mask_gen = MaskingMatrixGenerator(random_seed=mask_seed)
    mask_input = np.concatenate([u_star, v_star])
    widths_mask = None  # Let generator decide
    P_M = mask_gen.generate_probability_matrix(
        X=np.zeros((J, K)),  # dummy, not used for L>0
        U=mask_input.reshape(-1, 1),
        V=np.zeros((1,)),
        L=mask_L,
        widths=widths_mask
    ).reshape(J, K)

    # 6. Generate multiple big mask matrices
    big_mask_matrices = []
    for mask_idx in range(num_big_masks):
        # Generate binary mask from probability matrix
        if mask_seed is not None:
            np.random.seed(mask_seed + mask_idx)  # Different seed for each mask
        random_values = np.random.random(P_M.shape)
        binary_mask = (random_values < P_M).astype(int)
        
        # Create big mask matrix by tiling
        big_mask_matrix = np.zeros((N*J, T*K))
        for j in range(J):
            for k in range(K):
                row_start = j*N
                row_end = (j+1)*N
                col_start = k*T
                col_end = (k+1)*T
                if binary_mask[j, k] == 1:
                    # Convert masking matrix to binary (threshold at 0.5)
                    binary_masking_matrix = (masking_matrices[j, k] > 0.5).astype(int)
                    big_mask_matrix[row_start:row_end, col_start:col_end] = binary_masking_matrix
                else:
                    big_mask_matrix[row_start:row_end, col_start:col_end] = 0
        big_mask_matrices.append(big_mask_matrix)

    return big_data_matrix, big_mask_matrices, P_M, sampled_pairs

def test_hierarchical_generation():
    """
    Example usage and test for run_hierarchical_generation.
    Generates random data and masking matrices, runs the pipeline, and prints output shapes and summaries.
    """
    J, K, N, T = 3, 4, 5, 6  # Example sizes
    B = 5  # Number of pairs to sample
    num_big_masks = 3  # Number of big mask matrices to generate
    np.random.seed(0)
    # Generate random data and masking matrices
    data_matrices = np.random.randn(J, K, N, T)
    # Generate binary masking matrices (threshold at 0.5)
    masking_matrices = (np.random.rand(J, K, N, T) > 0.5).astype(int)
    # Run hierarchical generation
    big_data_matrix, big_mask_matrices, P_M, sampled_pairs = run_hierarchical_generation(
        data_matrices,
        masking_matrices,
        B=B,
        data_L=2,
        mask_L=2,
        num_big_masks=num_big_masks,
        data_seed=123,
        mask_seed=456,
        pair_seed=789
    )
    print("Big data matrix shape:", big_data_matrix.shape)
    print(f"Generated {len(big_mask_matrices)} big mask matrices, each shape: {big_mask_matrices[0].shape}")
    print("Probability mask (P_M) shape:", P_M.shape)
    print("Sampled pairs:", sampled_pairs)
    print("P_M (rounded):\n", np.round(P_M, 3))
    print(f"\nGenerated {num_big_masks} big mask matrices:")
    for mask_idx, big_mask_matrix in enumerate(big_mask_matrices):
        print(f"\nMask {mask_idx + 1}:")
        print("Binary values (0s and 1s):", np.unique(big_mask_matrix))
        print("Block sums:")
        for j in range(J):
            for k in range(K):
                block_sum = big_mask_matrix[j*N:(j+1)*N, k*T:(k+1)*T].sum()
                print(f"  Block ({j},{k}) sum: {block_sum:.2f}")

if __name__ == "__main__":
    test_hierarchical_generation() 