import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkGenerator:
    """
    A neural network generator that creates a network with specified architecture
    and returns probability outputs.
    
    Mathematical Structure:
    - Input: x ∈ R^p
    - L layers with widths W = [w_1, w_2, ..., w_L]
    - Layer l: z^(l) = W^(l) * a^(l-1) + b^(l)
    - Activation: a^(l) = σ(z^(l)) for l < L
    - Output: softmax(z^(L)) for probabilities
    """
    
    def __init__(self, input_size_p, layer_sizes_L, widths_W, 
                 activation='relu', random_seed=None, verbose=False):
        """
        Initialize the neural network generator.
        
        Parameters:
        - input_size_p (int): Size of input vector x (p)
        - layer_sizes_L (int): Number of layers (L)
        - widths_W (list): Width of each layer [w_1, w_2, ..., w_L]
        - activation (str): Activation function ('relu', 'sigmoid', 'tanh')
        - random_seed (int): Random seed for reproducibility
        - verbose (bool): If True, print layer info and architecture
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.p = input_size_p
        self.L = layer_sizes_L
        self.W = widths_W
        self.activation = activation
        self.verbose = verbose
        
        # Validate inputs
        if len(widths_W) != layer_sizes_L:
            raise ValueError(f"Number of widths ({len(widths_W)}) must equal number of layers ({layer_sizes_L})")
        
        # Generate weight matrices and bias vectors
        self.weights, self.biases = self._generate_parameters()
        
        if self.verbose:
            print("Neural Network Generated:")
            print(f"Input size (p): {self.p}")
            print(f"Number of layers (L): {self.L}")
            print(f"Layer widths (W): {self.W}")
            print(f"Activation function: {self.activation}")
            print(f"Architecture: {self.p} → {' → '.join(map(str, self.W))}")
            print("-" * 50)
        
    def _generate_parameters(self):
        """
        Generate random weight matrices and bias vectors for all layers.
        
        Mathematical formulation:
        W^(l) ∈ R^(w_l × w_{l-1}) for l = 1, ..., L
        b^(l) ∈ R^(w_l) for l = 1, ..., L
        
        where w_0 = p (input size)
        """
        weights = {}
        biases = {}
        
        # Layer dimensions: [input_size, width_1, width_2, ..., width_L]
        layer_dims = [self.p] + self.W
        
        for layer in range(1, self.L + 1):
            input_dim = layer_dims[layer-1]  # Previous layer size
            output_dim = layer_dims[layer]   # Current layer size
            
            # Xavier/He initialization based on activation function
            if self.activation == 'relu':
                # He initialization: better for ReLU
                std = np.sqrt(2.0 / input_dim)
            else:
                # Xavier initialization: better for sigmoid/tanh
                std = np.sqrt(1.0 / input_dim)
            
            # Generate weight matrix W^(l)
            weights[f'W{layer}'] = np.random.normal(0, std, (output_dim, input_dim))
            
            # Generate bias vector b^(l) (initialized to small random values)
            biases[f'b{layer}'] = np.random.normal(0, 0.01, (output_dim, 1))
            
            if self.verbose:
                print(f"Layer {layer}: W{layer} shape = {weights[f'W{layer}'].shape}, "
                      f"b{layer} shape = {biases[f'b{layer}'].shape}")
        
        return weights, biases
    
    def _relu(self, z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)
    
    def _sigmoid(self, z):
        """Sigmoid activation: 1 / (1 + exp(-z))"""
        # Clip to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
    
    def _tanh(self, z):
        """Tanh activation: tanh(z)"""
        return np.tanh(z)
    
    def _softmax(self, z):
        """
        Softmax activation for probability output.
        
        Mathematical formulation:
        softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def _apply_activation(self, z, activation_type):
        """Apply the specified activation function"""
        if activation_type == 'relu':
            return self._relu(z)
        elif activation_type == 'sigmoid':
            return self._sigmoid(z)
        elif activation_type == 'tanh':
            return self._tanh(z)
        elif activation_type == 'softmax':
            return self._softmax(z)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    def forward_pass(self, x, return_intermediates=False):
        """
        Perform forward pass through the network.
        
        Mathematical formulation:
        a^(0) = x
        For l = 1, 2, ..., L-1:
            z^(l) = W^(l) @ a^(l-1) + b^(l)
            a^(l) = σ(z^(l))
        
        For output layer L:
            z^(L) = W^(L) @ a^(L-1) + b^(L)
            a^(L) = softmax(z^(L))  [probabilities]
        
        Parameters:
        - x: Input vector of shape (p, 1) or (p,)
        - return_intermediates: If True, return all intermediate activations
        
        Returns:
        - probabilities: Output probabilities from softmax
        - intermediates: (optional) Dictionary of all layer outputs
        """
        # Ensure x is a column vector
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        if x.shape[0] != self.p:
            raise ValueError(f"Input size {x.shape[0]} doesn't match expected size {self.p}")
        
        # Store intermediate values
        activations = {'a0': x}  # Input layer
        pre_activations = {}     # Pre-activation values
        
        # Forward pass through all layers
        current_activation = x
        
        for layer in range(1, self.L + 1):
            # Linear transformation: z^(l) = W^(l) @ a^(l-1) + b^(l)
            z = np.dot(self.weights[f'W{layer}'], current_activation) + self.biases[f'b{layer}']
            pre_activations[f'z{layer}'] = z
            
            # Apply activation function
            if layer < self.L:
                # Hidden layers: use specified activation
                a = self._apply_activation(z, self.activation)
            else:
                # Output layer: use softmax for probabilities
                a = self._apply_activation(z, 'softmax')
            
            activations[f'a{layer}'] = a
            current_activation = a
        
        # Output probabilities
        probabilities = activations[f'a{self.L}']
        
        if return_intermediates:
            intermediates = {
                'pre_activations': pre_activations,
                'activations': activations
            }
            return probabilities, intermediates
        else:
            return probabilities
    
    def generate_output(self, x, verbose=True):
        """
        Main function: Generate probability output for input x.
        
        Parameters:
        - x: Input data vector of size p
        - verbose: If True, print detailed information
        
        Returns:
        - probabilities: Probability vector (sums to 1)
        """
        if verbose:
            print("\nForward Pass:")
            print(f"Input x shape: {x.shape if hasattr(x, 'shape') else len(x)}")
        
        # Get probabilities and intermediate values
        probabilities, intermediates = self.forward_pass(x, return_intermediates=True)
        
        if verbose:
            # Print layer-by-layer computation
            print("\nLayer-by-layer computation:")
            activations = intermediates['activations']
            pre_activations = intermediates['pre_activations']
            
            for layer in range(1, self.L + 1):
                z = pre_activations[f'z{layer}']
                a = activations[f'a{layer}']
                
                if layer < self.L:
                    activation_name = self.activation
                else:
                    activation_name = 'softmax'
                
                print(f"Layer {layer}:")
                print(f"  Pre-activation z^({layer}) shape: {z.shape}")
                print(f"  Post-activation a^({layer}) ({activation_name}) shape: {a.shape}")
                print(f"  Sample values: {a.flatten()[:5]}")  # Show first 5 values
        
        return probabilities.flatten()
    
    def visualize_network(self):
        """Visualize the network architecture and weights"""
        fig, axes = plt.subplots(1, self.L, figsize=(4*self.L, 6))
        
        if self.L == 1:
            axes = [axes]
        
        for layer in range(1, self.L + 1):
            ax = axes[layer-1]
            weight_matrix = self.weights[f'W{layer}']
            
            # Plot weight matrix as heatmap
            im = ax.imshow(weight_matrix, cmap='RdBu', aspect='auto')
            ax.set_title(f'Layer {layer} Weights\nShape: {weight_matrix.shape}')
            ax.set_xlabel(f'Input from Layer {layer-1}')
            ax.set_ylabel(f'Neurons in Layer {layer}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Neural Network Weights Visualization\nArchitecture: {self.p} → {" → ".join(map(str, self.W))}')
        plt.tight_layout()
        plt.show()
    
    def print_parameters(self):
        """Print all network parameters"""
        print("\nNetwork Parameters:")
        print("=" * 50)
        
        total_params = 0
        
        for layer in range(1, self.L + 1):
            W = self.weights[f'W{layer}']
            b = self.biases[f'b{layer}']
            
            layer_params = W.size + b.size
            total_params += layer_params
            
            print(f"Layer {layer}:")
            print(f"  Weight matrix W^({layer}): {W.shape} = {W.size} parameters")
            print(f"  Bias vector b^({layer}): {b.shape} = {b.size} parameters")
            print(f"  Layer total: {layer_params} parameters")
            print()
        
        print(f"Total network parameters: {total_params}")

def demo_usage():
    """Demonstrate the neural network generator with examples"""
    print("NEURAL NETWORK GENERATOR DEMO")
    print("=" * 60)
    
    # Example 1: Small network
    print("\nExample 1: Small Network")
    print("-" * 30)
    
    # Input: 5-dimensional vector
    # Architecture: 5 → 10 → 8 → 3 (3 output classes)
    nn1 = NeuralNetworkGenerator(
        input_size_p=5,
        layer_sizes_L=3,
        widths_W=[10, 8, 3],
        activation='relu',
        random_seed=42
    )
    
    # Generate random input
    x1 = np.random.randn(5)
    print(f"\nInput: {x1}")
    
    # Get output probabilities
    probs1 = nn1.generate_output(x1, verbose=True)
    print(f"\nOutput probabilities: {probs1}")
    print(f"Sum of probabilities: {np.sum(probs1):.6f}")
    print(f"Predicted class: {np.argmax(probs1)}")
    
    # Example 2: Deeper network
    print("\n\nExample 2: Deeper Network")
    print("-" * 30)
    
    # Input: 20-dimensional vector
    # Architecture: 20 → 50 → 30 → 20 → 10 → 5
    nn2 = NeuralNetworkGenerator(
        input_size_p=20,
        layer_sizes_L=5,
        widths_W=[50, 30, 20, 10, 5],
        activation='tanh',
        random_seed=123
    )
    
    # Generate random input
    x2 = np.random.randn(20)
    
    # Get output probabilities
    probs2 = nn2.generate_output(x2, verbose=False)
    print(f"Output probabilities: {probs2}")
    print(f"Sum of probabilities: {np.sum(probs2):.6f}")
    
    # Visualize the networks
    print("\nVisualizing network weights...")
    nn1.visualize_network()
    
    # Print parameter counts
    nn1.print_parameters()
    nn2.print_parameters()
    
    return nn1, nn2

def test_batch_processing():
    """Test the network with multiple inputs (batch processing)"""
    print("\n\nBATCH PROCESSING TEST")
    print("=" * 40)
    
    # Create a small network
    nn = NeuralNetworkGenerator(
        input_size_p=4,
        layer_sizes_L=2,
        widths_W=[8, 3],
        activation='sigmoid',
        random_seed=42
    )
    
    # Test with multiple inputs
    n_samples = 5
    X = np.random.randn(4, n_samples)  # 5 samples of 4-dimensional data
    
    print(f"Processing {n_samples} samples...")
    print(f"Input batch shape: {X.shape}")
    
    # Process each sample
    for i in range(n_samples):
        x_i = X[:, i]
        probs_i = nn.generate_output(x_i, verbose=False)
        print(f"Sample {i+1}: Probabilities = {probs_i}, Predicted class = {np.argmax(probs_i)}")

def mathematical_verification():
    """Verify the mathematical correctness of the implementation"""
    print("\n\nMATHEMATICAL VERIFICATION")
    print("=" * 50)
    
    # Create a simple 2-layer network for manual verification
    nn = NeuralNetworkGenerator(
        input_size_p=2,
        layer_sizes_L=2,
        widths_W=[3, 2],
        activation='relu',
        random_seed=0
    )
    
    # Input vector
    x = np.array([1.0, -0.5])
    print(f"Input x: {x}")
    
    # Manual forward pass verification
    print("\nManual computation:")
    
    # Layer 1
    W1 = nn.weights['W1']
    b1 = nn.biases['b1']
    z1 = W1 @ x.reshape(-1, 1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    
    print(f"W1 shape: {W1.shape}")
    print(f"b1 shape: {b1.shape}")
    print(f"z1 = W1 @ x + b1 = {z1.flatten()}")
    print(f"a1 = ReLU(z1) = {a1.flatten()}")
    
    # Layer 2
    W2 = nn.weights['W2']
    b2 = nn.biases['b2']
    z2 = W2 @ a1 + b2
    
    # Softmax
    z2_shifted = z2 - np.max(z2)
    exp_z2 = np.exp(z2_shifted)
    a2 = exp_z2 / np.sum(exp_z2)
    
    print(f"W2 shape: {W2.shape}")
    print(f"b2 shape: {b2.shape}")
    print(f"z2 = W2 @ a1 + b2 = {z2.flatten()}")
    print(f"a2 = softmax(z2) = {a2.flatten()}")
    print(f"Sum of probabilities: {np.sum(a2)}")
    
    # Compare with network output
    probs = nn.generate_output(x, verbose=False)
    print(f"\nNetwork output: {probs}")
    print(f"Manual computation: {a2.flatten()}")
    print(f"Difference: {np.abs(probs - a2.flatten())}")
    print(f"Match: {np.allclose(probs, a2.flatten())}")

def custom_input_demo():
    """Allow user to specify custom inputs for testing"""
    print("\n\nCUSTOM INPUT DEMO")
    print("=" * 50)
    
    # Create a network
    input_size = 5
    nn = NeuralNetworkGenerator(
        input_size_p=input_size,
        layer_sizes_L=3,
        widths_W=[10, 8, 3],
        activation='relu',
        random_seed=42
    )
    
    print(f"Created network with input size: {input_size}")
    print(f"Architecture: {input_size} → 10 → 8 → 3")
    
    # Example 1: All zeros
    x1 = np.zeros(input_size)
    print(f"\nTest 1 - All zeros: {x1}")
    probs1 = nn.generate_output(x1, verbose=False)
    print(f"Output: {probs1}, Predicted class: {np.argmax(probs1)}")
    
    # Example 2: All ones
    x2 = np.ones(input_size)
    print(f"\nTest 2 - All ones: {x2}")
    probs2 = nn.generate_output(x2, verbose=False)
    print(f"Output: {probs2}, Predicted class: {np.argmax(probs2)}")
    
    # Example 3: Custom values
    x3 = np.array([1.0, -1.0, 0.5, -0.5, 2.0])
    print(f"\nTest 3 - Custom values: {x3}")
    probs3 = nn.generate_output(x3, verbose=False)
    print(f"Output: {probs3}, Predicted class: {np.argmax(probs3)}")
    
    # Example 4: Random input
    x4 = np.random.randn(input_size)
    print(f"\nTest 4 - Random input: {x4}")
    probs4 = nn.generate_output(x4, verbose=False)
    print(f"Output: {probs4}, Predicted class: {np.argmax(probs4)}")
    
    return nn

def interactive_input_test():
    """Interactive function to test custom inputs"""
    print("\n\nINTERACTIVE INPUT TEST")
    print("=" * 50)
    
    # Get network parameters from user
    try:
        input_size = int(input("Enter input size (default 5): ") or "5")
        num_layers = int(input("Enter number of layers (default 3): ") or "3")
        
        # Generate random layer widths
        layer_widths = np.random.randint(2, 21, num_layers)
        print(f"Generated layer widths: {layer_widths}")
        
        activation = input("Enter activation function (relu/sigmoid/tanh, default relu): ") or "relu"
        
        # Create network
        nn = NeuralNetworkGenerator(
            input_size_p=input_size,
            layer_sizes_L=num_layers,
            widths_W=layer_widths.tolist(),
            activation=activation,
            random_seed=42
        )
        
        print(f"\nNetwork created! Architecture: {input_size} → {' → '.join(map(str, layer_widths))}")
        
        # Test custom inputs
        while True:
            print(f"\nEnter {input_size} numbers separated by spaces (or 'quit' to exit):")
            user_input = input().strip()
            
            if user_input.lower() == 'quit':
                break
                
            try:
                # Parse input
                values = [float(x) for x in user_input.split()]
                if len(values) != input_size:
                    print(f"Error: Expected {input_size} values, got {len(values)}")
                    continue
                    
                x = np.array(values)
                print(f"Input: {x}")
                
                # Get output
                probs = nn.generate_output(x, verbose=False)
                print(f"Output probabilities: {probs}")
                print(f"Predicted class: {np.argmax(probs)}")
                print(f"Sum of probabilities: {np.sum(probs):.6f}")
                
            except ValueError:
                print("Error: Please enter valid numbers")
                
    except ValueError as e:
        print(f"Error: {e}")
        return None
    
    return nn

if __name__ == "__main__":
    # Run demonstrations
    print("Running standard demos...")
    nn1, nn2 = demo_usage()
    test_batch_processing()
    mathematical_verification()
    
    # Run custom input demos
    print("\n" + "="*60)
    print("CUSTOM INPUT TESTING")
    print("="*60)
    
    # Demo with predefined inputs
    custom_input_demo()
    
    # Interactive input testing (uncomment to use)
    # print("\nStarting interactive input test...")
    # interactive_input_test()
    
    print("\nTo test your own inputs, you can:")
    print("1. Use custom_input_demo() for predefined examples")
    print("2. Use interactive_input_test() for interactive input")
    print("3. Import NeuralNetworkGenerator and create your own inputs")