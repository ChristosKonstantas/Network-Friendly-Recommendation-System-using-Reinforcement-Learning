import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import warnings
import random

class ModelBased:
    def __init__(self, env):
        self.env = env
        self.trans_prob_array = env.transition_probability_matrix()
        self.value_evolution = None  # Placeholder for storing the evolution of the value function

    def value_iteration(self, gamma, epsilon):
        t = 0
        V = (np.zeros(len(self.env.state_space), dtype=np.float64))
        value_evolution = np.zeros((0, len(self.env.state_space)),
                                dtype=np.float64)  # 2D array to store the evolution of the Value function
        value_evolution = np.vstack((value_evolution, V))
        while True:
            Q = np.zeros((len(self.env.state_space), len(self.env.action_space)), dtype=np.float64)

            for s in range(len(self.env.state_space) - 1):
                for action_index in range(len(self.env.action_space)):
                    for s_next in range(len(self.env.state_space)):
                        # Bellman expectation equation (we use the min operator because we are in the min-cost setting)
                        Q[s][action_index] += self.trans_prob_array[s, action_index, s_next] * (
                                self.env.reward_function(s, s_next, action_index) + gamma * V[
                            s_next])

            if np.max(np.abs(V - np.min(Q, axis=1))) < epsilon:  # norm-1
                break

            V = np.min(Q,
                    axis=1)  # Bellman optimality equation to minimize Q over all actions and take the optimal state values
            value_evolution = np.vstack(
                (value_evolution, V))  # Append the current state Value function to value_evolution to plot it

            t += 1  # Increment the number of iterations

        pi = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[
            s]  # Return the optimal policy given the optimal value function
        print('Converged after %d iterations' % t)

        return V, pi, Q, value_evolution
    
    def policy_iteration(self, gamma, epsilon):
        t = 0
        value_evolution = np.zeros((0, len(self.env.state_space)),
                                dtype=np.float64)  # 2D array to store the evolution of the Value function
        
        random_actions = np.random.choice(list(range(len(self.env.action_space))),
                                        len(self.env.state_space))  # start with random actions for each state
        
        #cached_costs = np.where(self.env.cached_matrix == 0, -1, self.env.cached_matrix)

        pi = lambda s: {s: a for s, a in enumerate(random_actions)}[
            s]  # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)

        while True:
            old_pi = {s: pi(s) for s in range(len(self.env.state_space))}  # keep the old policy to compare with new
            # evaluate latest policy --> you receive its converged value function
            V = self.policy_evaluation(pi, gamma, epsilon)
            value_evolution = np.vstack((value_evolution, V))  # append the latest value function to value_evolution
            pi, Q = self.policy_improvement(V, gamma)  # improve the policy
            t += 1

            if old_pi == {s: pi(s) for s in range(len(
                    self.env.state_space))}:  # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
                break
        print('converged after %d iterations' % t)  # keep track of the number of (outer) iterations to converge
        return V, pi, Q, value_evolution

    def policy_evaluation(self, pi, gamma, epsilon):
        t = 0
        prev_V = np.zeros(len(self.env.state_space))
        # Repeat all value sweeps until convergence
        while True:
            V = np.zeros(len(self.env.state_space))

            for s in range(len(self.env.state_space)):
                if s == len(self.env.state_space) - 1:
                    continue
                else:
                    for s_next in range(len(self.env.state_space)):
                        V[s] += self.trans_prob_array[s, pi(s), s_next] * (
                                self.env.reward_function(s, s_next, pi(s)) + gamma * prev_V[s_next])
            if np.max(np.abs(prev_V - V)) < epsilon:
                break
            prev_V = V.copy()
            t += 1

            return V


    def policy_improvement(self, V, gamma):
        Q = np.zeros((len(self.env.state_space), len(self.env.action_space)), dtype=np.float64)

        for s in range(len(self.env.state_space)):

            if s != len(self.env.state_space) - 1:
                for action_index in range(len(self.env.action_space)):
                    for s_next in range(len(self.env.state_space)):
                        Q[s][action_index] += self.trans_prob_array[s, action_index, s_next] * (
                                self.env.reward_function(s, s_next,action_index) + gamma * V[
                            s_next])
        new_pi = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]

        return new_pi, Q

    


class ModelFree:
    def __init__(self, env):
        self.env = env
        self.Q = None  # Placeholder for Q-table
        self.cached_costs = None  # Placeholder for cached costs
        
    def SARSA(self, num_episodes, learning_rate, discount_factor):
        num_states = len(self.env.state_space)
        num_actions = len(self.env.action_space)
        Q = np.zeros((num_states, num_actions))
        t_s = np.ones((len(self.env.state_space), 1))

        # SARSA algorithm
        for _ in tqdm(range(num_episodes), desc = 'SARSA is running'):
            # Initialize the state
            s = np.random.choice(self.env.state_space)  # randomly initialize the state from p = np.full(k+1, 1 / (k+1)) (uniform)
            while s != len(self.env.state_space) - 1:  # repeat until we sample a terminal state from the pmf above
                epsilon = t_s[s] ** (-1 / 3)
                # Choose an action using epsilon-greedy policy
                action_space_s = [i for i in self.env.action_space if s not in i]
                action_space_s_ind = [list(self.env.combinations_dict.keys())[list(self.env.combinations_dict.values()).index(act)] for
                                    act
                                    in action_space_s]
                if np.random.rand() < epsilon:
                    # Explore
                    action = np.random.choice(action_space_s_ind,
                                            p=np.full(len(action_space_s_ind), 1 / (len(action_space_s_ind)),
                                                        dtype=np.float16))  # Explore
                else:
                    # Exploit
                    Q_values = Q[s][action_space_s_ind]  # Filter Q-values for legal actions
                    action = action_space_s_ind[np.argmin(Q_values)]  # Select action with min Q-value

                # Perform the action and observe the next state and reward
                # s_next = np.random.choice(state_space, p=trans_prob_array[s][action][:])

                # Q-learning doesn't know 'a' but this is does not depend on the user's behavior on random episodes
                s_next, recom = self.env.step(action)

                # Choose a good next action using epsilon-greedy policy
                action_space_snext = [i for i in self.env.action_space if s not in i]
                action_space_snext_ind = [list(self.env.combinations_dict.keys())[list(self.env.combinations_dict.values()).index(act)] for
                                        act
                                        in action_space_snext]

                if np.random.rand() < epsilon:
                    # Explore
                    action_next = np.random.choice(action_space_snext_ind,
                                                p=np.full(len(action_space_snext_ind), 1 / (len(action_space_s_ind)),
                                                            dtype=np.float16))  # Explore
                else:
                    # Exploit
                    Q_values = Q[s_next][action_space_snext_ind]  # Filter Q-values for legal actions
                    action_next = action_space_snext_ind[np.argmin(Q_values)]  # Select action with min Q-value

                # Update Q-value using the Q-learning update rule

                target_estimate = self.env.reward_function(s, s_next, action) + discount_factor * Q[
                    s_next, action_next]
                Q[s, action] += learning_rate * (target_estimate - Q[s, action])

                s = s_next
                action = action_next
                t_s[s] += 1

        pi = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]

        return Q, pi
        
    def Q_learning_scheduled(self, num_episodes, learning_rate_schedule, gamma):
        
        # Initialize the Q-table
        num_states = len(self.env.state_space)
        num_actions = len(self.env.action_space)
        Q = np.zeros((num_states, num_actions))  # Q-table initialized with zeros
        cost_per_round = np.zeros((num_episodes,))  # Track total rewards for each episode and state
        cached_costs = np.where(self.env.cached_matrix == 0, -1, self.env.cached_matrix)

        # Q-learning algorithm
        for episode in tqdm(range(num_episodes), desc ='Q_Learning running'):
            learning_rate = learning_rate_schedule[episode]  # Select learning rate for current episode
            total_cost = 0  # Track total reward for the episode
            # Initialize the state
            t_s = np.ones((len(self.env.state_space), 1))  # Visit count for each state initialized with ones
            s = np.random.choice(self.env.state_space)  # Randomly initialize the state from the state space
            while s != len(self.env.state_space) - 1:  # Repeat until a terminal state is reached
                epsilon = t_s[s] ** (-1 / 3)  # Calculate exploration probability based on visit count

                # Choose an action using epsilon-greedy policy
                action_space_s = [i for i in self.env.action_space if s not in i]  # Available actions in current state
                action_space_s_ind = [list(self.env.combinations_dict.keys())[list(self.env.combinations_dict.values()).index(act)] for
                                    act in action_space_s]  # Indices of available actions in current state
                if np.random.rand() < epsilon:
                    # Explore: Randomly select an action from available actions
                    action = np.random.choice(action_space_s_ind,
                                            p=np.full(len(action_space_s_ind), 1 / (len(action_space_s_ind)),
                                                        dtype=np.float16))
                else:
                    # Exploit: Select action with the minimum Q-value for the current state
                    Q_values = Q[s][action_space_s_ind]  # Filter Q-values for legal actions
                    action = action_space_s_ind[np.argmin(Q_values)]  # Select action with min Q-value

                # Perform the action and observe the next state and reward
                s_next, _ = self.env.step(action)  # Determine next state based on current state and action
                reward = self.env.reward_function(s, s_next, action)  # Calculate the reward


                # Update Q-value using the Q-learning update rule
                Q[s, action] += learning_rate * (reward + gamma * np.min(Q[s_next, :]) - Q[s, action])

                s = s_next  # Transition to the next state
                t_s[s] += 1  # Increment visit count for the next state

            pi_temp = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]
            cost_per_round[episode] = self.env.get_expected_cost_per_round(pi_temp, total_cost, cached_costs)





        # Plot the average expected cached cost for all states
        plt.plot(np.arange(num_episodes), cost_per_round)
        plt.xlabel("Episode")
        plt.ylabel("Expected Cost")
        plt.title("Q-Learning: Expected Cost for all states")
        plt.text(num_episodes-50000, 1, r"cost $=-1$ for a cached content and $+1$ otherwise", fontsize=8, color="black")
        plt.show()

        # Define the policy function
        pi = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]

        return Q, pi, cost_per_round   

    def Q_learning(self, num_episodes, learning_rate, gamma):
        # Initialize the Q-table
        num_states = len(self.env.state_space)
        num_actions = len(self.env.action_space)
        Q = np.zeros((num_states, num_actions))
        t_s = np.ones((len(self.env.state_space), 1))
        # Q-learning algorithm
        for _ in range(num_episodes):
            # Initialize the state
            s = np.random.choice(self.env.state_space)  # randomly initialize the state from p = np.full(k+1, 1 / (k+1)) (uniform)
            while s != len(self.env.state_space) - 1:  # repeat until we sample a terminal state from the pmf above
                epsilon = t_s[s] ** (-1 / 3)
                # Choose an action using epsilon-greedy policy
                action_space_s = [i for i in self.env.action_space if s not in i]
                action_space_s_ind = [list(self.env.combinations_dict.keys())[list(self.env.combinations_dict.values()).index(act)] for
                                    act
                                    in action_space_s]
                if np.random.rand() < epsilon:
                    # Explore
                    action = np.random.choice(action_space_s_ind,
                                            p=np.full(len(action_space_s_ind), 1 / (len(action_space_s_ind)),
                                                        dtype=np.float16))  # Explore
                else:
                    # Exploit
                    Q_values = Q[s][action_space_s_ind]  # Filter Q-values for legal actions
                    action = action_space_s_ind[np.argmin(Q_values)]  # Select action with min Q-value

                # Perform the action and observe the next state and reward
                # s_next = np.random.choice(state_space, p=trans_prob_array[s][action][:])

                # Q-learning doesn't know 'a' but this is does not depend on the user's behavior on random episodes
                s_next, _ = self.env.step(action)

                # Update Q-value using the Q-learning update rule
                # s_next = np.random.choice(state_space, p=trans_prob_array[s][action][:]) # Oracle transition
                Q[s, action] += learning_rate * (
                        self.env.reward_function(s, s_next, action) + gamma * np.min(
                    Q[s_next, :]) - Q[s, action])

                s = s_next
                t_s[s] += 1
        pi = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]

        return Q, pi
    
    def meta_train(self, num_episodes, initial_learning_rate, gamma):
        learning_rate_schedule = np.ones(num_episodes) * initial_learning_rate
        learning_rate = initial_learning_rate
        performance_improvements = np.zeros(num_episodes)
        threshold = 0.1  # Initial threshold value
        increase_factor = 1.1  # Initial increase factor
        decrease_factor = 0.9  # Initial decrease factor
        performance_improvement = 0.0
        # Perform meta-training
        for episode in tqdm(range(num_episodes), desc= 'meta train is running '):
            # Call the Q-learning function with the current learning rate
            Q, _ = self.Q_learning(1, learning_rate_schedule[episode], gamma)

            # Determine the performance improvement based on Q-values or other metrics
            if episode > 0:
                prev_Q, _ = self.Q_learning(1, learning_rate_schedule[episode - 1],
                                            gamma)
                performance_improvement = np.mean(np.abs(Q - prev_Q))
                performance_improvements = np.append(performance_improvements, performance_improvement)

                # Adjust the learning rate based on the performance improvement
                if performance_improvement > threshold:
                    learning_rate *= increase_factor
                else:
                    learning_rate *= decrease_factor

                # Update the threshold as the mean of the previous 5 performance improvements
                threshold = np.mean(performance_improvements[-5:])

            # Update the learning rate schedule
            learning_rate_schedule = np.append(learning_rate_schedule, learning_rate)

        ''' 
        # Plot the learning rate schedule and performance improvements
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(num_episodes), learning_rate_schedule)
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_episodes), performance_improvements)
        plt.xlabel('Episode')
        plt.ylabel('Performance Improvement')


        plt.tight_layout()
        plt.show()
        '''

        return learning_rate_schedule
    
    def pick_minQ_action(self, state, actions, Q):
        Q_values = Q[state][actions]  # Filter Q-values for legal actions
        min_action = actions[np.argmin(Q_values)]  # Select action with min Q-value

        return min_action
    
    
class RL_Utils:
    def __init__(self, env):
        self.env = env
        
    def plot_q_values(self, Q, str_title):
        num_actions_per_bin = int(math.comb(len(self.env.state_space), len(self.env.state_space) - 1) / len(self.env.state_space))
        # Define the ranges for state and action spaces
        state_space = range(len(self.env.state_space))
        num_actions = len(self.env.action_space)

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
        plt.yticks(range(num_complete_bins), range(len(self.env.action_space))) # range(len(action_space)) converts the action_space range to a list of integers

        # Show the plot
        plt.show()

    @staticmethod
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
        
    @staticmethod
    def generate_random_seeds(self,num_seeds):
        random_seeds = random.sample(range(num_seeds * 10), num_seeds)
        return random_seeds
    
    @staticmethod
    def generate_random_policy(self, seed):
        np.random.seed(seed)
        random_policy = np.zeros(len(self.state_space) - 1, dtype=int)
        for s in range(len(self.state_space) - 1):
            random_policy[s] = np.random.randint(0, len(self.action_space))
        return random_policy

