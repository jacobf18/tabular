import numpy as np

mar_sequential_bandit_config = {
    'algorithm': 'epsilon_greedy',
    'pooling': False, 
    'epsilon': 0.4, 
    'epsilon_decay': 0.99, 
    'random_seed': 42
}
# config3 = {'algorithm': 'ucb', 'pooling': False, 'c': 1.5, 'ucb_variant': 'ucb_tuned'}
# config5 = {'algorithm': 'thompson_sampling', 'pooling': False, 'prior_std': 2.0, 'shrinkage_factor': 0.8}
# config7 = {'algorithm': 'gradient_bandit', 'pooling': False, 'alpha': 0.1, 'softmax_temperature': 1.0}

# # With pooling
# config2 = {'algorithm': 'epsilon_greedy', 'pooling': True, 'epsilon': 0.4, 'epsilon_decay': 0.99, 'pooling_weight': 0.4, 'update_frequency': 5, 'random_seed': 42}
# config4 = {'algorithm': 'ucb', 'pooling': True, 'c': 1.5, 'ucb_variant': 'ucb_tuned', 'pooling_weight': 0.4, 'update_frequency': 5}
# config6 = {'algorithm': 'thompson_sampling', 'pooling': True, 'prior_std': 2.0, 'shrinkage_factor': 0.8, 'pooling_weight': 0.4, 'update_frequency': 5}
# config8 = {'algorithm': 'gradient_bandit', 'pooling': True, 'alpha': 0.1, 'softmax_temperature': 1.0, 'pooling_weight': 0.4, 'update_frequency': 5}


class UnifiedBandit:
    """
    Unified bandit algorithm class with simplified configuration
    
    Supports 4 algorithms: epsilon_greedy, ucb, thompson_sampling, gradient_bandit
    Each can be used with or without intelligent pooling across units
    """
    
    def __init__(self, config):
        """
        Config Dictionary Structure:
        ---------------------------
        Required:
            algorithm : str - 'epsilon_greedy', 'ucb', 'thompson_sampling', 'gradient_bandit'
            pooling : bool - Whether to enable intelligent pooling across units
            
        Optional (with defaults):
            noise_std : float (default 1.0) - Standard deviation for reward noise
            noise_type : str (default 'gaussian') - 'gaussian', 'uniform', 'student_t'
            random_seed : int (default None) - Random seed for reproducibility
            
        Algorithm-Specific Parameters:
        -----------------------------
        Epsilon-Greedy:
            epsilon : float (default 0.3) - exploration probability
            epsilon_decay : float or None (default None) - decay rate per step
            
        UCB:
            c : float (default 1.0) - confidence parameter
            ucb_variant : str (default 'ucb1') - 'ucb1' or 'ucb_tuned'
            
        Thompson Sampling:
            prior_std : float (default 1.0) - prior standard deviation
            shrinkage_factor : float (default 1.0) - shrinkage toward prior
            
        Gradient Bandit:
            alpha : float (default 0.1) - learning rate
            softmax_temperature : float (default 1.0) - exploration temperature
            
        Pooling Parameters (only used if pooling=True):
        ----------------------------------------------
            pooling_weight : float (default 0.3) - weight for population data
            update_frequency : int (default 5) - steps between population updates
        """
        
        # Validate config
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        # Extract core configuration
        self.algorithm = config.get('algorithm', 'epsilon_greedy')
        self.pooling = config.get('pooling', False)
        self.noise_std = config.get('noise_std', 1.0)
        self.noise_type = config.get('noise_type', 'gaussian')
        
        # Validate algorithm
        valid_algorithms = ['epsilon_greedy', 'ucb', 'thompson_sampling', 'gradient_bandit']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Use {valid_algorithms}")
        
        # Validate noise type
        valid_noise_types = ['gaussian', 'uniform', 'student_t']
        if self.noise_type not in valid_noise_types:
            raise ValueError(f"Unknown noise_type: {self.noise_type}. Use {valid_noise_types}")
        
        # Set random seed
        random_seed = config.get('random_seed')
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Algorithm-specific parameters
        if self.algorithm == 'epsilon_greedy':
            self.epsilon = config.get('epsilon', 0.3)
            self.epsilon_decay = config.get('epsilon_decay', None)
            self.current_epsilon = self.epsilon  # Track current epsilon for decay
            
        elif self.algorithm == 'ucb':
            self.c = config.get('c', 1.0)
            self.ucb_variant = config.get('ucb_variant', 'ucb1')
            
        elif self.algorithm == 'thompson_sampling':
            self.prior_std = config.get('prior_std', 1.0)
            self.shrinkage_factor = config.get('shrinkage_factor', 1.0)
            self.prior_precision = 1.0 / (self.prior_std ** 2)
            self.likelihood_precision = 1.0 / (self.noise_std ** 2)
            
        elif self.algorithm == 'gradient_bandit':
            self.alpha = config.get('alpha', 0.1)
            self.softmax_temperature = config.get('softmax_temperature', 1.0)
        
        # Pooling parameters (only used if pooling=True)
        if self.pooling:
            self.pooling_weight = config.get('pooling_weight', 0.3)
            self.update_frequency = config.get('update_frequency', 5)
        
        # State variables (initialized in fit)
        self.rows = None
        self.cols = None
        self.assignment_matrix = None
        self.all_experiences = []
        self.unit_states = {}
        self.population_states = {}
    
    def create_arm_rewards(self, reward_matrix):
        """Create separate reward vectors with specified noise type"""
        rows, cols = reward_matrix.shape
        arm_0_rewards = np.zeros((rows, cols))
        arm_1_rewards = np.zeros((rows, cols))
        
        for row in range(rows):
            base_values = reward_matrix[row, :]
            
            if self.noise_type == 'gaussian':
                noise_vector = np.random.normal(0, self.noise_std, 2 * cols)
                
            elif self.noise_type == 'uniform':
                a = self.noise_std * np.sqrt(3)  # Match standard deviation
                noise_vector = np.random.uniform(-a, a, 2 * cols)
                
            elif self.noise_type == 'student_t':
                # Heavy-tailed noise with df=3 for moderate heavy tails
                noise_vector = np.random.standard_t(3, 2 * cols) * self.noise_std
                
            else:
                raise ValueError(f"Unknown noise_type: {self.noise_type}. Use 'gaussian', 'uniform', or 'student_t'")
            
            # Split noise and add to base values
            arm_0_rewards[row, :] = base_values + noise_vector[:cols]
            arm_1_rewards[row, :] = base_values + noise_vector[cols:]
        
        return arm_0_rewards, arm_1_rewards
    
    def initialize_states(self, rows, cols):
        """Initialize algorithm and pooling state variables"""
        self.rows = rows
        self.cols = cols
        
        # Algorithm-specific state initialization
        if self.algorithm in ['epsilon_greedy', 'ucb']:
            self.unit_states = {
                'arm_counts': np.zeros((rows, 2)),
                'arm_rewards': np.zeros((rows, 2))
            }
        
        elif self.algorithm == 'thompson_sampling':
            self.unit_states = {
                'posterior_precision': np.full((rows, 2), self.prior_precision),
                'posterior_mean': np.zeros((rows, 2))
            }
        
        elif self.algorithm == 'gradient_bandit':
            self.unit_states = {
                'H': np.zeros((rows, 2)),  # Action preferences
                'baseline': np.zeros(rows),
                'total_rewards': np.zeros(rows),
                'total_steps': np.zeros(rows)
            }
        
        # Pooling state initialization
        if self.pooling:
            self.population_states = {
                'pop_arm_counts': np.zeros(2),
                'pop_arm_rewards': np.zeros(2),
                'pop_variance': np.ones(2),
                'last_update_time': 0
            }
    
    def select_arm(self, unit, time):
        """Select arm for given unit at given time"""
        if not self.pooling:
            return self._select_arm_no_pooling(unit, time)
        else:
            return self._select_arm_with_pooling(unit, time)
    
    def _select_arm_no_pooling(self, unit, time):
        """Arm selection without pooling"""
        
        if self.algorithm == 'epsilon_greedy':
            arm_counts = self.unit_states['arm_counts'][unit]
            arm_rewards = self.unit_states['arm_rewards'][unit]
            
            # Apply epsilon decay if specified
            if self.epsilon_decay is not None and time > 0:
                self.current_epsilon = max(self.current_epsilon * self.epsilon_decay, 0.01)
            
            if np.random.random() < self.current_epsilon or min(arm_counts) == 0:
                return np.random.randint(0, 2)
            else:
                avg_rewards = arm_rewards / np.maximum(arm_counts, 1)
                return np.argmax(avg_rewards)
        
        elif self.algorithm == 'ucb':
            arm_counts = self.unit_states['arm_counts'][unit]
            arm_rewards = self.unit_states['arm_rewards'][unit]
            
            if min(arm_counts) == 0:
                return 0 if arm_counts[0] == 0 else 1
            else:
                avg_rewards = arm_rewards / arm_counts
                
                if self.ucb_variant == 'ucb1':
                    confidence = self.c * np.sqrt(np.log(time + 1) / arm_counts)
                elif self.ucb_variant == 'ucb_tuned':
                    # UCB-Tuned uses variance estimates
                    variance_est = np.maximum(avg_rewards * (1 - avg_rewards), 1e-6)
                    confidence = self.c * np.sqrt(np.log(time + 1) / arm_counts * 
                                                np.minimum(0.25, variance_est))
                
                ucb_values = avg_rewards + confidence
                return np.argmax(ucb_values)
        
        elif self.algorithm == 'thompson_sampling':
            posterior_precision = self.unit_states['posterior_precision'][unit]
            posterior_mean = self.unit_states['posterior_mean'][unit]
            
            posterior_var = 1.0 / posterior_precision
            sampled_rewards = np.array([
                np.random.normal(posterior_mean[0], np.sqrt(max(posterior_var[0], 1e-6))),
                np.random.normal(posterior_mean[1], np.sqrt(max(posterior_var[1], 1e-6)))
            ])
            return np.argmax(sampled_rewards)
        
        elif self.algorithm == 'gradient_bandit':
            H = self.unit_states['H'][unit]
            
            # Apply temperature scaling
            scaled_H = H / self.softmax_temperature
            exp_H = np.exp(scaled_H - np.max(scaled_H))  # Numerical stability
            probabilities = exp_H / np.sum(exp_H)
            return np.random.choice(2, p=probabilities)
    
    def _select_arm_with_pooling(self, unit, time):
        """Arm selection with pooling"""
        
        # Get unit's individual experience
        unit_experiences = [(t, a, r) for (u, t, a, r) in self.all_experiences if u == unit]
        
        # Need minimum data for pooling to be effective
        if len(self.all_experiences) < 20:
            return self._select_arm_no_pooling(unit, time)
        
        if self.algorithm == 'epsilon_greedy':
            return self._epsilon_greedy_pooled(unit, unit_experiences, time)
        elif self.algorithm == 'ucb':
            return self._ucb_pooled(unit, unit_experiences, time)
        elif self.algorithm == 'thompson_sampling':
            return self._thompson_pooled(unit, unit_experiences)
        elif self.algorithm == 'gradient_bandit':
            return self._gradient_pooled(unit, unit_experiences)
    
    def _epsilon_greedy_pooled(self, unit, unit_experiences, time):
        """Epsilon-greedy with pooling"""
        unit_counts = np.zeros(2)
        unit_rewards = np.zeros(2)
        
        for t, a, r in unit_experiences:
            unit_counts[a] += 1
            unit_rewards[a] += r
        
        # Combine individual + population data
        pop_counts = self.population_states['pop_arm_counts']
        pop_rewards = self.population_states['pop_arm_rewards']
        
        combined_counts = unit_counts + self.pooling_weight * pop_counts
        combined_rewards = unit_rewards + self.pooling_weight * pop_rewards
        
        # Apply epsilon decay
        if self.epsilon_decay is not None and time > 0:
            self.current_epsilon = max(self.current_epsilon * self.epsilon_decay, 0.01)
        
        if np.random.random() < self.current_epsilon or min(combined_counts) == 0:
            return np.random.randint(0, 2)
        else:
            avg_rewards = combined_rewards / np.maximum(combined_counts, 1)
            return np.argmax(avg_rewards)
    
    def _ucb_pooled(self, unit, unit_experiences, time):
        """UCB with pooling"""
        unit_counts = np.zeros(2)
        unit_rewards = np.zeros(2)
        
        for t, a, r in unit_experiences:
            unit_counts[a] += 1
            unit_rewards[a] += r
        
        # Combine with population data
        pop_counts = self.population_states['pop_arm_counts']
        pop_rewards = self.population_states['pop_arm_rewards']
        pop_variance = self.population_states['pop_variance']
        
        combined_counts = unit_counts + self.pooling_weight * pop_counts
        combined_rewards = unit_rewards + self.pooling_weight * pop_rewards
        
        if min(combined_counts) == 0:
            return 0 if combined_counts[0] == 0 else 1
        else:
            avg_rewards = combined_rewards / combined_counts
            
            if self.ucb_variant == 'ucb1':
                confidence = self.c * np.sqrt((pop_variance + np.log(time + 1)) / combined_counts)
            elif self.ucb_variant == 'ucb_tuned':
                variance_est = np.maximum(avg_rewards * (1 - avg_rewards), pop_variance)
                confidence = self.c * np.sqrt(np.log(time + 1) / combined_counts * 
                                            np.minimum(0.25, variance_est))
            
            ucb_values = avg_rewards + confidence
            return np.argmax(ucb_values)
    
    def _thompson_pooled(self, unit, unit_experiences):
        """Thompson Sampling with Bayesian pooling"""
        unit_counts = np.zeros(2)
        unit_rewards = np.zeros(2)
        
        for t, a, r in unit_experiences:
            unit_counts[a] += 1
            unit_rewards[a] += r
        
        # Use population mean as prior
        pop_mean = self.population_states['pop_arm_rewards'] / np.maximum(
            self.population_states['pop_arm_counts'], 1)
        
        # Bayesian update with shrinkage
        shrunk_prior_precision = self.prior_precision * self.shrinkage_factor
        posterior_precision = shrunk_prior_precision + unit_counts * self.likelihood_precision
        
        posterior_mean = (shrunk_prior_precision * pop_mean + 
                         self.likelihood_precision * unit_rewards) / posterior_precision
        
        posterior_var = 1.0 / posterior_precision
        sampled_rewards = np.array([
            np.random.normal(posterior_mean[0], np.sqrt(max(posterior_var[0], 1e-6))),
            np.random.normal(posterior_mean[1], np.sqrt(max(posterior_var[1], 1e-6)))
        ])
        return np.argmax(sampled_rewards)
    
    def _gradient_pooled(self, unit, unit_experiences):
        """Gradient bandit with pooled baseline"""
        if len(unit_experiences) == 0:
            return np.random.randint(0, 2)
        
        # Use population baseline
        all_rewards = [r for (u, t, a, r) in self.all_experiences]
        pop_baseline = np.mean(all_rewards) if all_rewards else 0
        
        # Build preferences from unit's experience
        H = np.zeros(2)
        for t, a, r in unit_experiences:
            reward_diff = r - pop_baseline
            H[a] += self.alpha * reward_diff
        
        # Apply temperature scaling
        scaled_H = H / self.softmax_temperature
        exp_H = np.exp(scaled_H - np.max(scaled_H))
        probabilities = exp_H / np.sum(exp_H)
        return np.random.choice(2, p=probabilities)
    
    def update_unit(self, unit, chosen_arm, observed_reward, time):
        """Update unit's parameters after observing reward"""
        
        if self.algorithm in ['epsilon_greedy', 'ucb']:
            self.unit_states['arm_counts'][unit, chosen_arm] += 1
            self.unit_states['arm_rewards'][unit, chosen_arm] += observed_reward
        
        elif self.algorithm == 'thompson_sampling':
            old_precision = self.unit_states['posterior_precision'][unit, chosen_arm]
            old_mean = self.unit_states['posterior_mean'][unit, chosen_arm]
            
            # Bayesian update with shrinkage
            shrunk_likelihood = self.likelihood_precision * self.shrinkage_factor
            self.unit_states['posterior_precision'][unit, chosen_arm] += shrunk_likelihood
            self.unit_states['posterior_mean'][unit, chosen_arm] = (
                (old_precision * old_mean + shrunk_likelihood * observed_reward) / 
                self.unit_states['posterior_precision'][unit, chosen_arm]
            )
        
        elif self.algorithm == 'gradient_bandit':
            # Update baseline
            self.unit_states['total_rewards'][unit] += observed_reward
            self.unit_states['total_steps'][unit] += 1
            self.unit_states['baseline'][unit] = (
                self.unit_states['total_rewards'][unit] / 
                self.unit_states['total_steps'][unit]
            )
            
            # Update preferences
            reward_diff = observed_reward - self.unit_states['baseline'][unit]
            H = self.unit_states['H'][unit]
            
            # Apply temperature to probabilities for gradient update
            scaled_H = H / self.softmax_temperature
            exp_H = np.exp(scaled_H - np.max(scaled_H))
            probabilities = exp_H / np.sum(exp_H)
            
            for arm in range(2):
                if arm == chosen_arm:
                    self.unit_states['H'][unit, arm] += self.alpha * reward_diff * (1 - probabilities[arm])
                else:
                    self.unit_states['H'][unit, arm] -= self.alpha * reward_diff * probabilities[arm]
    
    def update_population(self, time):
        """Update population-level parameters for pooling"""
        if not self.pooling or len(self.all_experiences) < 10:
            return
        
        # Update population statistics
        arm_counts = np.zeros(2)
        arm_rewards = np.zeros(2)
        arm_values = [[], []]
        
        for unit, t, arm, reward in self.all_experiences:
            arm_counts[arm] += 1
            arm_rewards[arm] += reward
            arm_values[arm].append(reward)
        
        self.population_states['pop_arm_counts'] = arm_counts
        self.population_states['pop_arm_rewards'] = arm_rewards
        
        # Update population variance for confidence bounds
        for arm in range(2):
            if len(arm_values[arm]) > 1:
                self.population_states['pop_variance'][arm] = max(np.var(arm_values[arm]), 0.01)
        
        self.population_states['last_update_time'] = time
    
    def fit(self, reward_matrix):
        """
        Run the bandit algorithm on the reward matrix
        
        Parameters:
        -----------
        reward_matrix : np.ndarray
            Shape (rows, cols) where rows are units and cols are time points
        
        Returns:
        --------
        assignment_matrix : np.ndarray
            Shape (rows, cols) with 0/1 assignments
        arm_0_rewards : np.ndarray
            Reward structure for arm 0
        arm_1_rewards : np.ndarray
            Reward structure for arm 1
        """
        
        rows, cols = reward_matrix.shape
        
        # Create arm rewards with noise
        arm_0_rewards, arm_1_rewards = self.create_arm_rewards(reward_matrix)
        
        # Initialize states
        self.initialize_states(rows, cols)
        
        # Initialize assignment matrix and experience log
        self.assignment_matrix = np.zeros((rows, cols), dtype=int)
        self.all_experiences = []
        
        # Reset current epsilon for decay
        if hasattr(self, 'epsilon'):
            self.current_epsilon = self.epsilon
        
        # Main learning loop
        for time in range(cols):
            for unit in range(rows):
                
                # Select arm
                chosen_arm = self.select_arm(unit, time)
                
                # Observe reward
                if chosen_arm == 0:
                    observed_reward = arm_0_rewards[unit, time]
                else:
                    observed_reward = arm_1_rewards[unit, time]
                
                # Store experience
                self.all_experiences.append((unit, time, chosen_arm, observed_reward))
                
                # Update unit parameters
                self.update_unit(unit, chosen_arm, observed_reward, time)
                
                # Record assignment
                self.assignment_matrix[unit, time] = chosen_arm
            
            # Update population parameters if pooling enabled
            if self.pooling and (time + 1) % self.update_frequency == 0:
                self.update_population(time)
        
        return self.assignment_matrix, arm_0_rewards, arm_1_rewards
    
    def get_config(self):
        """Return current configuration"""
        config = {
            'algorithm': self.algorithm,
            'pooling': self.pooling,
            'noise_std': self.noise_std,
            'noise_type': self.noise_type
        }
        
        # Add algorithm-specific configs
        if self.algorithm == 'epsilon_greedy':
            config.update({
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay
            })
        elif self.algorithm == 'ucb':
            config.update({
                'c': self.c,
                'ucb_variant': self.ucb_variant
            })
        elif self.algorithm == 'thompson_sampling':
            config.update({
                'prior_std': self.prior_std,
                'shrinkage_factor': self.shrinkage_factor
            })
        elif self.algorithm == 'gradient_bandit':
            config.update({
                'alpha': self.alpha,
                'softmax_temperature': self.softmax_temperature
            })
        
        # Add pooling configs
        if self.pooling:
            config.update({
                'pooling_weight': self.pooling_weight,
                'update_frequency': self.update_frequency
            })
        
        return config
    
    