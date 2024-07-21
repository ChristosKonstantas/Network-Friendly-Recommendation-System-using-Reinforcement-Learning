import math
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import warnings


def create_symmetric_matrix(k):
    matrix = np.zeros((k, k))  # Initialize matrix with zeros

    for i in range(k):
        for j in range(i + 1, k):
            m = random.random()  # Assign random value between 0 and 1
            matrix[i][j] = m
            matrix[j][i] = m  # Assign the same value symmetrically

    return matrix


def print_matrix(matrix):
    for row in matrix:
        print('[', end='')
        for i, element in enumerate(row):
            print('{:.2f}'.format(element), end='')
            if i != len(row) - 1:
                print(', ', end='')
        print(']')


def random_ind_cache_matrix(C, k):
    cached_matrix = np.ones((k, k))
    np.fill_diagonal(cached_matrix, 0)  # Fill diagonal elements with zeros
    for i in range(k):
        j_elem = np.random.choice([idx for idx in range(k) if idx != i], size=C, replace=False)
        for j in range(len(j_elem)):
            cached_matrix[i][j_elem[j]] = 0
    return cached_matrix



def N_highest_values(U, k, N):
    # The instruction below sorts the elements in each row of the matrix U in ascending order and returns
    # the corresponding indices. Then, [:, -N:] selects the last N indices (corresponding to the highest values)
    # from each row.

    # Find the indices of the N highest values in each row
    sorted_indices = np.argsort(U, axis=1)[:,
                     -N:]  # sorts the elements in each row of the matrix U in ascending order and returns the corresponding indices. Then, [:, -N:] selects the last N indices (corresponding to the highest values) from each row.

    # Create the new matrix with zeros
    high_matrix = np.zeros((k, N))

    # Fill the new matrix with the N highest values from the original matrix
    for i in range(k):
        high_matrix[i] = U[i, sorted_indices[i]]
    return [high_matrix, sorted_indices]


def dict_of_combinations(k, N):
    # Generate all possible tuples of combinations of N distinct contents out of k contents
    combinations = list(itertools.combinations(range(k), N))
    # print(combinations[0]) # gives (0,1)

    # Exclude combinations with repeated elements to ensure there are no same item recommendations in a batch
    distinct_combinations = [comb for comb in combinations if len(set(comb)) == N]

    # Create a dictionary with distinct combinations as keys and index as values
    return {idx: list(comb) for idx, comb in enumerate(distinct_combinations)}


def plot_value_evolution(value_evolution, states, iterations, str_title):
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




def plot_q_values(Q, state_space, action_space, str_title):
    num_actions_per_bin = int(math.comb(len(state_space), len(state_space) - 1) / len(state_space))
    # Define the ranges for state and action spaces
    state_space = range(len(state_space))
    num_actions = len(action_space)

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
    plt.yticks(range(num_complete_bins), range(len(action_space))) # range(len(action_space)) converts the action_space range to a list of integers

    # Show the plot
    plt.show()

def plot_cached_matrix(matrix):
    # Convert the matrix to a NumPy array
    matrix = np.array(matrix)

    # Create a binary image where 0 represents 'cached' and 1 represents 'non-cached'
    binary_image = np.where(matrix == 0, 0, 1)

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


def generate_random_seeds(num_seeds):
    random_seeds = random.sample(range(num_seeds * 10), num_seeds)
    return random_seeds

def has_duplicates(lst):
    seen = set()
    for item in lst:
        if item in seen:
            return True
        seen.add(item)
    return False

def generate_random_policy(state_space, action_space, seed):
    np.random.seed(seed)
    random_policy = np.zeros(len(state_space) - 1, dtype=int)
    for s in range(len(state_space) - 1):
        random_policy[s] = np.random.randint(0, len(action_space))
    return random_policy

def calculate_cost_metrics(user_sessions, cached_matrix, state_space):
    """
    Calculates the cost metrics for a given policy
    """
    cost_metrics = []
    for session in user_sessions:
        cost_metrics.append(calculate_cost_metric(session, cached_matrix, state_space))
    return cost_metrics

def calculate_cost_metric(session, cached_matrix, state_space):

    cost_metric = 0
    for i in range(len(session)-1):
        if session[i] == len(state_space)-1 or session[i+1] == len(state_space)-1:
            break
        else:
            cost_metric += cached_matrix[session[i], session[i + 1]]
    expected_cost = cost_metric/len(session)
    return expected_cost

def get_expected_cost_per_round(action_space, state_space, pi, total_cost, cached_costs):
    for s in range(len(state_space) - 1):
        batch = action_space[pi(s)]
        for i in batch:
            if i == s or i == len(state_space) - 1:
                total_cost += 0
            else:
                total_cost += cached_costs[s, i]

    return total_cost / (len(state_space) - 1)
