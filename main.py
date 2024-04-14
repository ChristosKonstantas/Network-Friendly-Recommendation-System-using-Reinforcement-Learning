from list_dict_func import *
import numpy as np
import math
import time
from tqdm import tqdm


# This will be done for every now state s and every action
def transition_probabilities(s, s_next, w_s, q, a):
    if s > k - 1:
        raise Exception('s=k=' + str(s), 'which is not in the catalog of contents but a terminal state')
    else:
        if s_next == s:
            return 0
        elif s_next == k:
            return q
        else:
            if relevant(w_s, U, s):
                if s_next in w_s:
                    return (1 - q) * (a / N + (1 - a) / (
                            k - 1))  # we use k-1 because we exclude the item for the probability (s_next == s)
                else:
                    return (1 - q) * (1 - a) / (
                            k - 1)  # we use k-1 because we exclude the item for the probability (s_next == s)
            else:
                return (1 - q) * (
                        1 / (k - 1))  # we use k-1 because we exclude the item for the probability (s_next == s)


# Serves for reducing expected cost due to non-cached contents, but it is not so good for ensuring user satisfaction
# Although the transition probabilities give a little greater probability that user clicks one of the recommendations

def transition_probability_matrix(state_space, action_space):
    trans_prob_array = np.zeros((len(state_space) - 1, len(action_space), len(state_space)))
    print('shape is', trans_prob_array.shape)
    for s in range(len(state_space) - 1):
        action_space_s = [i for i in action_space if s not in i]
        action_space_s_ind = [list(combinations_dict.keys())[list(combinations_dict.values()).index(action)] for action
                            in action_space_s]
        for action_index in action_space_s_ind:
            for s_next in range(len(state_space)):
                trans_prob_array[s, action_index, s_next] = transition_probabilities(s, s_next,
                                                                                    action_space[action_index], q, a)
    return trans_prob_array

def relevant(w, U, user_now_watches):
    if user_now_watches < k:
        return all(U[item, user_now_watches] > u_min for item in w)
    else:
        raise Exception('user_now_watches=k=' + str(user_now_watches),
                        'which is not in the catalog of contents but a terminal state')
        
def reward_function(s, s_next, cached_matrix, w_s):
    # 1-> bad, -1-> good
    if s <= k - 1:
        if s_next in w_s:
            if s_next == s:
                return 1_000_000  # it should never get here but just in case we have a bug in the code and we get here we
                # want to penalize it a lot
            elif s_next == k:  # if the user enters the terminal state from a content state
                return 1
            else:
                if cached_matrix[s, s_next] == 0:
                    return -1
                return 1
        else:
            return -1
    else:
        raise Exception('s=k=' + str(s), 'which is not in the catalog of contents but a terminal state')

def surfing_user(state_space, action_space, action):
    p = np.random.rand()
    recom = 0
    if p < q:  # q ligopithano an o user einai picky
        s_next = k
    else:
        u = np.random.rand()
        if u < a:
            s_next = np.random.choice(action_space[action])  # Pick one of the recommended items at random
            recom = 1
        else:
            s_next = np.random.choice(range(len(state_space) - 1))  # Pick a random item from the catalog
    return s_next, recom

def simulate_user_sessions(policy, state_space, action_space, num_of_sessions):
    sessions = []
    np.random.seed(None)
    for i in range(num_of_sessions):
        session = []
        state = np.random.randint(0, len(state_space) - 1)
        session.append(state)
        while True:
            action = policy(state)
            next_state, recom = surfing_user(state_space, action_space, action)
            session.append(next_state)
            if next_state == len(state_space) - 1:
                break
            state = next_state
        sessions.append(session)
    return sessions

def value_iteration(state_space, action_space, cached_matrix, gamma=1.0, epsilon=1e-10):
    trans_prob_array = transition_probability_matrix(state_space, action_space)
    t = 0
    V = (np.zeros(len(state_space), dtype=np.float64))
    value_evolution = np.zeros((0, len(state_space)),
                            dtype=np.float64)  # 2D array to store the evolution of the Value function
    value_evolution = np.vstack((value_evolution, V))
    while True:
        Q = np.zeros((len(state_space), len(action_space)), dtype=np.float64)

        for s in range(len(state_space) - 1):
            for action_index in range(len(action_space)):
                for s_next in range(len(state_space)):
                    # Bellman expectation equation (we use the min operator because we are in the min-cost setting)
                    Q[s][action_index] += trans_prob_array[s, action_index, s_next] * (
                            reward_function(s, s_next, cached_matrix, action_space[action_index]) + gamma * V[
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


def policy_evaluation(state_space, action_space, pi, trans_prob_array, gamma=1.0, epsilon=1e-10):
    t = 0
    prev_V = np.zeros(len(state_space))
    # Repeat all value sweeps until convergence
    while True:
        V = np.zeros(len(state_space))

        for s in range(len(state_space)):
            if s == len(state_space) - 1:
                continue
            else:
                for s_next in range(len(state_space)):
                    V[s] += trans_prob_array[s, pi(s), s_next] * (
                            reward_function(s, s_next, cached_matrix, action_space[pi(s)]) + gamma * prev_V[s_next])
        if np.max(np.abs(prev_V - V)) < epsilon:
            break
        prev_V = V.copy()
        t += 1

        return V


def policy_improvement(V, state_space, action_space, trans_prob_array, gamma=1.0, epsilon=1e-10):
    Q = np.zeros((len(state_space), len(action_space)), dtype=np.float64)

    for s in range(len(state_space)):

        if s != len(state_space) - 1:
            for action_index in range(len(action_space)):
                for s_next in range(len(state_space)):
                    Q[s][action_index] += trans_prob_array[s, action_index, s_next] * (
                            reward_function(s, s_next, cached_matrix, action_space[action_index]) + gamma * V[
                        s_next])
    new_pi = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]

    return new_pi, Q


def policy_iteration(state_space, action_space, trans_prob_array, gamma=1.0, epsilon=1e-10):
    t = 0
    value_evolution = np.zeros((0, len(state_space)),
                            dtype=np.float64)  # 2D array to store the evolution of the Value function

    total_cost = 0  # Track total reward for the episode
    random_actions = np.random.choice(list(range(len(action_space))),
                                    len(state_space))  # start with random actions for each state
    cached_costs = np.where(cached_matrix == 0, -1, cached_matrix)

    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[
        s]  # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)

    while True:
        old_pi = {s: pi(s) for s in range(len(state_space))}  # keep the old policy to compare with new
        # evaluate latest policy --> you receive its converged value function
        V = policy_evaluation(state_space, action_space, pi, trans_prob_array, gamma, epsilon)
        value_evolution = np.vstack((value_evolution, V))  # append the latest value function to value_evolution
        pi, Q = policy_improvement(V, state_space, action_space, trans_prob_array, gamma, epsilon)  # improve the policy
        t += 1

        if old_pi == {s: pi(s) for s in range(len(
                state_space))}:  # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
            break



    print('converged after %d iterations' % t)  # keep track of the number of (outer) iterations to converge
    return V, pi, Q, value_evolution


def Q_learning(state_space, action_space, num_episodes, learning_rate_schedule, discount_factor):
    # Initialize the Q-table
    num_states = len(state_space)
    num_actions = len(action_space)
    Q = np.zeros((num_states, num_actions))  # Q-table initialized with zeros
    cost_per_round = np.zeros((num_episodes,))  # Track total rewards for each episode and state
    cached_costs = np.where(cached_matrix == 0, -1, cached_matrix)

    # Q-learning algorithm
    for episode in tqdm(range(num_episodes), desc ='Q_Learning running'):
        learning_rate = learning_rate_schedule[episode]  # Select learning rate for current episode
        total_cost = 0  # Track total reward for the episode
        # Initialize the state
        t_s = np.ones((len(state_space), 1))  # Visit count for each state initialized with ones
        s = np.random.choice(state_space)  # Randomly initialize the state from the state space
        while s != len(state_space) - 1:  # Repeat until a terminal state is reached
            epsilon = t_s[s] ** (-1 / 3)  # Calculate exploration probability based on visit count

            # Choose an action using epsilon-greedy policy
            action_space_s = [i for i in action_space if s not in i]  # Available actions in current state
            action_space_s_ind = [list(combinations_dict.keys())[list(combinations_dict.values()).index(act)] for
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
            s_next, recom = surfing_user(state_space, action_space, action)  # Determine next state based on current state and action
            reward = reward_function(s, s_next, cached_matrix, action_space[action])  # Calculate the reward


            # Update Q-value using the Q-learning update rule
            Q[s, action] += learning_rate * (reward + discount_factor * np.min(Q[s_next, :]) - Q[s, action])

            s = s_next  # Transition to the next state
            t_s[s] += 1  # Increment visit count for the next state

        pi_temp = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]
        cost_per_round[episode] = get_expected_cost_per_round(action_space, state_space, pi_temp, total_cost, cached_costs)





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


def Q_learning_test(state_space, action_space, num_episodes, learning_rate, discount_factor):
    # Initialize the Q-table
    num_states = len(state_space)
    num_actions = len(action_space)
    Q = np.zeros((num_states, num_actions))
    t_s = np.ones((len(state_space), 1))
    # Q-learning algorithm
    for episode in range(num_episodes):
        # Initialize the state
        s = np.random.choice(state_space)  # randomly initialize the state from p = np.full(k+1, 1 / (k+1)) (uniform)
        while s != len(state_space) - 1:  # repeat until we sample a terminal state from the pmf above
            epsilon = t_s[s] ** (-1 / 3)
            # Choose an action using epsilon-greedy policy
            action_space_s = [i for i in action_space if s not in i]
            action_space_s_ind = [list(combinations_dict.keys())[list(combinations_dict.values()).index(act)] for
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
            s_next, recom = surfing_user(state_space, action_space, action)

            # Update Q-value using the Q-learning update rule
            # s_next = np.random.choice(state_space, p=trans_prob_array[s][action][:]) # Oracle transition
            Q[s, action] += learning_rate * (
                    reward_function(s, s_next, cached_matrix, action_space[action]) + discount_factor * np.min(
                Q[s_next, :]) - Q[s, action])

            s = s_next
            t_s[s] += 1
    pi = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]

    return Q, pi


def pick_minQ_action(state, actions, Q):
    Q_values = Q[state][actions]  # Filter Q-values for legal actions
    min_action = actions[np.argmin(Q_values)]  # Select action with min Q-value

    return min_action




def meta_train(state_space, action_space, num_episodes, initial_learning_rate, discount_factor):
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
        Q, _ = Q_learning_test(state_space, action_space, 1, learning_rate_schedule[episode], discount_factor)

        # Determine the performance improvement based on Q-values or other metrics
        if episode > 0:
            prev_Q, _ = Q_learning_test(state_space, action_space, 1, learning_rate_schedule[episode - 1],
                                        discount_factor)
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


def SARSA(state_space, action_space, num_episodes, learning_rate, discount_factor):
    # Initialize the Q-table
    num_states = len(state_space)
    num_actions = len(action_space)
    Q = np.zeros((num_states, num_actions))
    t_s = np.ones((len(state_space), 1))

    # SARSA algorithm
    for episode in tqdm(range(num_episodes), desc = 'SARSA is running'):
        # Initialize the state
        s = np.random.choice(state_space)  # randomly initialize the state from p = np.full(k+1, 1 / (k+1)) (uniform)
        while s != len(state_space) - 1:  # repeat until we sample a terminal state from the pmf above
            epsilon = t_s[s] ** (-1 / 3)
            # Choose an action using epsilon-greedy policy
            action_space_s = [i for i in action_space if s not in i]
            action_space_s_ind = [list(combinations_dict.keys())[list(combinations_dict.values()).index(act)] for
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
            s_next, recom = surfing_user(state_space, action_space, action)

            # Choose a good next action using epsilon-greedy policy
            action_space_snext = [i for i in action_space if s not in i]
            action_space_snext_ind = [list(combinations_dict.keys())[list(combinations_dict.values()).index(act)] for
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

            target_estimate = reward_function(s, s_next, cached_matrix, action_space[action]) + discount_factor * Q[
                s_next, action_next]
            Q[s, action] += learning_rate * (target_estimate - Q[s, action])

            s = s_next
            action = action_next
            t_s[s] += 1

    pi = lambda s: {s: a for s, a in enumerate(np.argmin(Q, axis=1))}[s]

    return Q, pi

if __name__ == '__main__':

    start = time.time()
    # Global variables
    # random.seed(42)
    u_min = 0.8
    q = 0.2
    a = 0.9  # for 1-q *and* ->relevance<-, user picks one of the N recommended contents with equal probability
    gamma = 1 - q
    k = 10 # number of contents
    N = 2  # recommendation batch size
    num_cached = 0.2  # out of 1, thus the num_cached*100%
    C = int(num_cached * k)
    print('k=', k)

    # a kxk symmetric matrix with values of main diagonal equal to zero and the other values are random between 0 and 1
    U = create_symmetric_matrix(k)
    # print('U=')
    # print_matrix(U)

    # create the matrix with the cached contents
    cached_matrix = random_ind_cache_matrix(C, k)
    plot_cached_matrix(cached_matrix)
    print('Cost matrix:\n', cached_matrix)
    print(
        '\n **------------------------------ Analysis of the MDP environment --------------------------------------------** \n')

    print(sum(cached_matrix[np.random.randint(0, k - 1)][:] == 1), "out of", k, 'contents are cached, approximately the',
        format((100 * (sum(cached_matrix[np.random.randint(0, k - 1)][:] == 1) / k)), ".2f"),
        '% of one random row of recommendations.')

    # UN_i is the set of  the highest N < k values of U related to content i=1,2,...,k
    # j_ind_UN is the set of the indices of the highest N < k values of U related to content i=1,2,...,k
    [UN_i, j_ind_UN] = N_highest_values(U, k, N)  # benchmark for user satisfaction
    # print_matrix(UN_i)
    # print(j_ind_UN)


    # Print total number of possible combinations <=> Print number of possible batches of recommendations
    print('We have', math.comb(k - 1, N), 'possible combinations of', N, 'contents out of', k,
        ' contents that are all the possible batches of recommendations if we use the batches as actions.')

    # Create a dictionary with the math.comb(k, N) possible combinations of N contents out of k contents
    # The keys of the dictionary are the possible combinations of N contents out of k contents
    # The values of the dictionary are the indices of the contents in each combination
    # For example, if k=4 and N=2, the dictionary will be:
    # {0: [0, 1], 1: [0, 2], 2: [0, 3], 3: [1, 2], 4: [1, 3], 5: [2, 3]}

    combinations_dict = dict_of_combinations(k, N)

    print(combinations_dict)
    # print("Number of all possible distinct batches of size N:", len(distinct_combinations), 'and
    # the number of total combinations is' , len(combinations) )
    # print("\nDistinct Combinations Dictionary:")
    # for comb, idx in combinations_dict.items():
    #    print(comb, ":", idx)

    #  -- Construction of the Markov Decision Process (MDP) -- #
    state_space = np.arange(0, k + 1)  # k+1 states: 0, ...,k where k is the terminal state
    print('The state space has length of', len(state_space))
    action_space = [comb for comb in combinations_dict.values()]
    print("\nState Space:", state_space)
    print("\nAction Space:", action_space)
    '''
    A = transition_probabilities(9, 0, [0, 1], q, a)
    print(A)
    B = transition_probabilities(9, 1, [0, 1], q, a)
    print(B)
    C = transition_probabilities(9, 2, [0, 1], q, a)
    print(C)
    D = transition_probabilities(9, 3, [0, 1], q, a)
    print(D)
    E = transition_probabilities(9, 4, [0, 1], q, a)
    print(E)
    F = transition_probabilities(9, 5, [0, 1], q, a)
    print(F)
    G = transition_probabilities(9, 6, [0, 1], q, a)
    print(G)
    H = transition_probabilities(9, 7, [0, 1], q, a)
    print(H)
    I = transition_probabilities(9, 8, [0, 1], q, a)
    print(I)
    J = transition_probabilities(9, 9, [0, 1], q, a)
    print(J)
    K = transition_probabilities(9, 10, [0, 1], q, a)
    print(K)

    print('Here', A + B + C + D + E + F + G + H + I + J + K)  # should be equal to 1
    '''


    

    ############ BENCHMARKING ############


    # *** (1) and (2) are model based methods *** #
    # (1) ---------- Execute the value iteration algorithm -------------- #
    startVI = time.time()
    V, pi, Q, value_evolution = value_iteration(state_space, action_space, cached_matrix, 0.8, 1e-5)
    endVI = time.time()
    # Calculate states and iterations
    states = np.arange(len(state_space))
    iterations = np.arange(value_evolution.shape[0])
    # Plot value evolution
    plot_value_evolution(value_evolution, states, iterations, 'Value Iteration')
    plot_q_values(Q, state_space, action_space, 'Value Iteration')
    print("Execution time for Value Iteration: ", endVI - startVI, "seconds \n")

    print('\n--- Value function for value iteration ---- \n')
    print(V)

    print('\n--- Q values for value iteration ---- \n')
    print(Q)

    print('\n--- Policy after value iteration ---- \n')
    for s in range(len(state_space) - 1):
        batch = action_space[pi(s)]
        print('\n If user watches content s= ', s, 'we recommend ', batch)
        print('\n Relevance between s and batch[0]: ', U[s, batch[0]])
        print('\n Relevance between s and batch[1]: ', U[s, batch[1]])
        print('\n Cache cost between s and batch[0]', cached_matrix[s, batch[0]])
        print('\n Cache cost between s and batch[1]', cached_matrix[s, batch[1]])
        print('\n-------------------------------------------\n')

    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')

    # (2) ---------- Execute the policy iteration algorithm -------------- #

    trans_prob_array = transition_probability_matrix(state_space, action_space)
    startPI = time.time()
    V2, pi2, Q2, valEvol = policy_iteration(state_space, action_space, trans_prob_array, 0.8, 1e-5)
    states = np.arange(len(state_space))
    iterations = np.arange(value_evolution.shape[0])
    plot_value_evolution(value_evolution, states, iterations, 'Policy Iteration')

    endPI = time.time()
    print("Execution time for Policy Iteration: ", endPI - startPI, "seconds \n")
    plot_q_values(Q2, state_space, action_space, 'Policy Iteration')
    print('\n--- Policy after policy iteration ---- \n')
    for s in range(len(state_space) - 1):
        batch = action_space[pi2(s)]
        print('\nIf user watches content s=', s, 'we recommend ', batch)
        print('\nRelevance between s and batch[0]: ', U[s, batch[0]])
        print('\nRelevance between s and batch[1]: ', U[s, batch[1]])
        print('\nCache cost between s and batch[0]', cached_matrix[s, batch[0]])
        print('\nCache cost between s and batch[1]', cached_matrix[s, batch[1]])
        print('\n----------------------------------------------------\n')

    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')


    pi_PolicyIteration = pi2

    # Step 4: Calculate the average cost metric for policy iteration after simulating 10 user sessions
    #PI_sessions = simulate_user_sessions(pi_PolicyIteration, state_space, action_space, 10)
    #PI_cost = np.sum(calculate_cost_metrics(PI_sessions, cached_matrix, state_space))/len(PI_sessions)


    # *** (3) and (4) are model free methods *** #
    # (3) ---------- Execute SARSA algorithm -------------- #
    startSARSA = time.time()
    Q_SARSA, pi_SARSA = SARSA(state_space, action_space, 100000, 0.1, gamma)
    endSARSA = time.time()
    print("Execution time for SARSA: ", endSARSA - startSARSA, "seconds \n")

    print('\n--- Policy after SARSA ---- \n')
    for s in range(len(state_space) - 1):
        batch = action_space[pi_SARSA(s)]
        print('\nIf user watches content s=', s, 'we recommend ', batch)
        print('\nRelevance between s and batch[0]: ', U[s, batch[0]])
        print('\nRelevance between s and batch[1]: ', U[s, batch[1]])
        print('\nCache cost between s and batch[0]', cached_matrix[s, batch[0]])
        print('\nCache cost between s and batch[1]', cached_matrix[s, batch[1]])
        print('\n----------------------------------------------------\n')

    plot_q_values(Q_SARSA, state_space, action_space, 'SARSA')

    print('\n---************************************** ---- \n')
    if all(pi_SARSA(s) == pi2(s) for s in range(len(state_space) - 1)):
        print('The policies are the same')



    # (4) ---------- Execute the Q-learning algorithm -------------- #

    episodesQ = 60000 # number of episodes
    init_learning_rate = 0.01 # initial learning rate
    startLearningRate = time.time()
    learning_rate_schedule = meta_train(state_space, action_space, episodesQ, init_learning_rate, gamma)
    endLearningRate = time.time()

    print("Execution time for learning rate schedule: ", endLearningRate - startLearningRate, "seconds \n")

    startQL = time.time()

    QQ, piQ, cost = Q_learning(state_space, action_space, episodesQ, learning_rate_schedule, gamma)


    print(learning_rate_schedule)

    endQL = time.time()
    print("Execution time for Q Learning: ", endQL - startQL, "seconds \n")

    print('\n--- Policy after Q learning ---- \n')
    for s in range(len(state_space) - 1):
        batch = action_space[piQ(s)]
        print('\nIf user watches content s=', s, 'we recommend ', batch)
        print('\nRelevance between s and batch[0]: ', U[s, batch[0]])
        print('\nRelevance between s and batch[1]: ', U[s, batch[1]])
        print('\nCache cost between s and batch[0]', cached_matrix[s, batch[0]])
        print('\nCache cost between s and batch[1]', cached_matrix[s, batch[1]])
        print('\n----------------------------------------------------\n')

    plot_q_values(QQ, state_space, action_space, 'Q Learning')

    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')

    if all(piQ(s) == pi2(s) for s in range(len(state_space) - 1)):
        print('The policies are the same with  Policy Iteration')