import math
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import warnings

class NFR_Environment:
    def __init__(self, k, N, num_cached, q, a, u_min):
        self.k = k
        self.N = N
        self.num_cached = num_cached
        self.q = q
        self.a = a
        self.u_min = u_min
        self.gamma = 1 - q
        self.C = int(num_cached * k)
        self.U = self.create_symmetric_matrix()
        self.cached_matrix = self.random_ind_cache_matrix()
        self.combinations_dict = self.dict_of_combinations()
        self.state_space = np.arange(0, k + 1)
        self.action_space = [comb for comb in self.combinations_dict.values()]

    def step(self, action):
        p = np.random.rand()
        recom = 0
        if p < self.q:  # q has less probability if user is picky
            s_next = self.k
        else:
            u = np.random.rand()
            if u < self.a:
                s_next = np.random.choice(self.action_space[action])  # Pick one of the N recommended items at random
                recom = 1
            else:
                s_next = np.random.choice(range(len(self.state_space) - 1))  # Pick a random item from the catalog
        return s_next, recom
    
    def create_symmetric_matrix(self):
        matrix = np.zeros((self.k, self.k))  # Initialize matrix with zeros

        for i in range(self.k):
            for j in range(i + 1, self.k):
                m = random.random()  # Assign random value between 0 and 1
                matrix[i][j] = m
                matrix[j][i] = m  # Assign the same value symmetrically

        return matrix
    
    def reward_function(self, s, s_next, action_index):
        w_s = self.action_space[action_index]
        # 1-> bad, -1-> good
        if s <= self.k - 1:
            if s_next in w_s:
                if s_next == s:
                    return 1_000_000  # it should never get here
                elif s_next == self.k:  # if the user enters the terminal state from a content state
                    return 1
                else:
                    if self.cached_matrix[s, s_next] == 0:
                        return -1
                    return 1
            else:
                return -1
        else:
            raise Exception('s=k=' + str(s), 'which is not in the catalog of contents but a terminal state')
        
    @staticmethod
    def print_matrix(matrix):
        for row in matrix:
            print('[', end='')
            for i, element in enumerate(row):
                print('{:.2f}'.format(element), end='')
                if i != len(row) - 1:
                    print(', ', end='')
            print(']')


    def random_ind_cache_matrix(self):
        cached_matrix = np.ones((self.k, self.k))
        np.fill_diagonal(cached_matrix, 0)  # Fill diagonal elements with zeros
        for i in range(self.k):
            j_elem = np.random.choice([idx for idx in range(self.k) if idx != i], size=self.C, replace=False)
            for j in range(len(j_elem)):
                cached_matrix[i][j_elem[j]] = 0
        return cached_matrix



    def N_highest_values(self):
        # The instruction below sorts the elements in each row of the matrix U in ascending order and returns
        # the corresponding indices. Then, [:, -N:] selects the last N indices (corresponding to the highest values)
        # from each row.

        # Find the indices of the N highest values in each row
        sorted_indices = np.argsort(self.U, axis=1)[:,
                        -self.N:]  # sorts the elements in each row of the matrix U in ascending order and returns the corresponding indices. Then, [:, -N:] selects the last N indices (corresponding to the highest values) from each row.

        # Create the new matrix with zeros
        high_matrix = np.zeros((self.k, self.N))

        # Fill the new matrix with the N highest values from the original matrix
        for i in range(self.k):
            high_matrix[i] = self.U[i, sorted_indices[i]]
        return [high_matrix, sorted_indices]


    def dict_of_combinations(self):
        # Generate all possible tuples of combinations of N distinct contents out of k contents
        combinations = list(itertools.combinations(range(self.k), self.N))
        # print(combinations[0]) # gives (0,1)

        # Exclude combinations with repeated elements to ensure there are no same item recommendations in a batch
        distinct_combinations = [comb for comb in combinations if len(set(comb)) == self.N]

        # Create a dictionary with distinct combinations as keys and index as values
        return {idx: list(comb) for idx, comb in enumerate(distinct_combinations)}
    
    def plot_cached_matrix(self):
        # Convert the matrix to a NumPy array
        self.cached_matrix = np.array(self.cached_matrix)

        # Create a binary image where 0 represents 'cached' and 1 represents 'non-cached'
        binary_image = np.where(self.cached_matrix == 0, 0, 1)

        # Increase figure size for better visibility of axis labels and title
        plt.figure(figsize=(8, 6))

        # Plot the binary image
        plt.imshow(binary_image, cmap='gray')

        # Add a title to the image
        plt.title('Cached Matrix', pad=20)

        # Add labels to the axes
        plt.xlabel('Item')
        plt.ylabel('Current State')

        # Set the tick locations and labels for the y-axis
        y_ticks = np.arange(0, binary_image.shape[0], 1)
        plt.yticks(y_ticks, y_ticks.astype(int))

        # Set the tick locations and labels for the x-axis
        x_ticks = np.arange(0, binary_image.shape[1], 1)
        plt.xticks(x_ticks, x_ticks.astype(int))

        # Create a legend for the color description
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color='white', label='non-cached: 1'),
                        plt.Rectangle((0, 0), 1, 1, color='black', label='cached: 0')]
        plt.legend(handles=legend_elements, loc='upper right')

        # Show the plot
        plt.show()

    def plot_value_evolution(self, value_evolution, states, iterations, str_title):
        # Create meshgrid arrays
        states_mesh, iterations_mesh = np.meshgrid(states, iterations)

        # Flatten the meshgrid arrays and value_evolution array for scatter plotting
        states_flat = states_mesh.flatten()
        iterations_flat = iterations_mesh.flatten()
        values_flat = value_evolution.flatten()

        # Define the colormap
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            cmap = plt.cm.get_cmap('rainbow', lut=None)

        # Reverse the colormap
        # Normalize the values for mapping to colormap
        value_min = np.min(values_flat)
        value_max = np.max(values_flat)
        normalized_values = (values_flat - value_min) / (value_max - value_min)

        # Create the figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the scatter points with color mapping
        scatter = ax.scatter(states_flat, iterations_flat, values_flat, c=normalized_values, cmap=cmap, edgecolor='black')

        # Set labels and title
        ax.set_xlabel('States')
        ax.set_ylabel('Iterations')
        ax.set_zlabel('Value')
        ax.set_title(f'Value Evolution in {str_title}')
        ax.view_init(elev=20, azim=120)

        # Create a scalar mappable for the colorbar
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(value_min, value_max), cmap=cmap)
        sm.set_array([])  # An empty array is required

        # Show the colorbar
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Value')

        # Show the plot
        plt.show()




    def plot_q_values(self, Q, str_title):
        num_actions_per_bin = int(math.comb(len(self.state_space), len(self.state_space) - 1) / len(self.state_space))
        # Define the ranges for state and action spaces
        state_space = range(len(self.state_space))
        num_actions = len(self.action_space)

        # Calculate the number of complete bins
        num_complete_bins = num_actions // num_actions_per_bin

        # Calculate the number of actions that fit into complete bins
        num_actions_to_keep = num_complete_bins * num_actions_per_bin

        # Truncate the Q-values to keep only the actions that fit into complete bins
        Q_truncated = Q[:, :num_actions_to_keep]

        # Create a grid of state-action pairs
        X, Y = np.meshgrid(state_space, range(num_complete_bins))

        # Reshape the Q-values to match the grid shape
        Q_values = Q_truncated.reshape((len(state_space), num_complete_bins, num_actions_per_bin))

        # Increase figure size for better visibility of action labels
        plt.figure(figsize=(10, 8))

        # Plot the Q-values as a heatmap
        plt.imshow(Q_values.transpose(1, 0, 2), cmap='tab20b', interpolation='nearest', aspect='auto')
        plt.colorbar()

        # Set labels and title
        plt.xlabel('State')
        plt.ylabel('Action Bin')
        plt.title(f'Q-Values Colormap (Actions per Bin = {num_actions_per_bin}) for {str_title}')

        # Set the tick labels to integers and update y-axis labels
        plt.yticks(range(num_complete_bins), range(len(self.action_space))) # range(len(action_space)) converts the action_space range to a list of integers

        # Show the plot
        plt.show()


    def generate_random_seeds(self,num_seeds):
        random_seeds = random.sample(range(num_seeds * 10), num_seeds)
        return random_seeds

    def has_duplicates(self, lst):
        seen = set()
        for item in lst:
            if item in seen:
                return True
            seen.add(item)
        return False

    def generate_random_policy(self, seed):
        np.random.seed(seed)
        random_policy = np.zeros(len(self.state_space) - 1, dtype=int)
        for s in range(len(self.state_space) - 1):
            random_policy[s] = np.random.randint(0, len(self.action_space))
        return random_policy

    def calculate_cost_metrics(self, user_sessions):
        """
        Calculates the cost metrics for a given policy
        """
        cost_metrics = []
        for session in user_sessions:
            cost_metrics.append(self.calculate_cost_metric(session, self.cached_matrix, self.state_space))
        return cost_metrics

    def calculate_cost_metric(self, session):

        cost_metric = 0
        for i in range(len(session)-1):
            if session[i] == len(self.state_space)-1 or session[i+1] == len(self.state_space)-1:
                break
            else:
                cost_metric += self.cached_matrix[session[i], session[i + 1]]
        expected_cost = cost_metric/len(session)
        return expected_cost
    
    def get_expected_cost_per_round(self, pi, total_cost, cached_costs):
        for s in range(len(self.state_space) - 1):
            batch = self.action_space[pi(s)]
            for i in batch:
                if i == s or i == len(self.state_space) - 1:
                    total_cost += 0
                else:
                    total_cost += cached_costs[s, i]

        return total_cost / (len(self.state_space) - 1)

    def simulate_user_sessions(self,policy, num_of_sessions):
        sessions = []
        np.random.seed(None)
        for i in range(num_of_sessions):
            session = []
            state = np.random.randint(0, len(self.state_space) - 1)
            session.append(state)
            while True:
                action = policy(state)
                next_state, recom = self.step(action)
                session.append(next_state)
                if next_state == len(self.state_space) - 1:
                    break
                state = next_state
            sessions.append(session)
        return sessions