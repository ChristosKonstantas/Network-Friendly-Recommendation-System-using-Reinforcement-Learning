import numpy as np
import math
import time
import matplotlib.pyplot as plt
from utils import NFR_Environment    
from rl_algorithms import ModelBased, ModelFree, RL_Utils


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
    
    # Network Friendly Recommendations environment
    env = NFR_Environment(k=k, N=N, num_cached=num_cached, q=q, a=a, u_min=u_min)
    C = int(num_cached * k)
    print('k=', k)

    # a kxk symmetric matrix with values of main diagonal equal to zero and the other values are random between 0 and 1
    U = env.create_symmetric_matrix()

    # create the matrix with the cached contents
    env.plot_cached_matrix()
    print('Cost matrix:\n', env.cached_matrix)
    print(
        '\n **------------------------------ Analysis of the MDP environment --------------------------------------------** \n')

    print(sum(env.cached_matrix[np.random.randint(0, k - 1)][:] == 1), "out of", k, 'contents are cached, approximately the',
        format((100 * (sum(env.cached_matrix[np.random.randint(0, k - 1)][:] == 1) / k)), ".2f"),
        '% of one random row of recommendations.')

    # UN_i is the set of  the highest N < k values of U related to content i=1,2,...,k
    # j_ind_UN is the set of the indices of the highest N < k values of U related to content i=1,2,...,k
    [UN_i, j_ind_UN] = env.N_highest_values()  # benchmark for user satisfaction
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

    combinations_dict = env.dict_of_combinations()
    print(combinations_dict)

    #  -- Construction of the Markov Decision Process (MDP) -- #
    print('The state space has length of', len(env.state_space))
    print("\nState Space:", env.state_space)
    print("\nAction Space:", env.action_space)
    
    '''
    A = env.transition_probabilities(9, 0, [0, 1], q, a)
    print(A)
    B = env.transition_probabilities(9, 1, [0, 1], q, a)
    print(B)
    C = env.transition_probabilities(9, 2, [0, 1], q, a)
    print(C)
    D = env.transition_probabilities(9, 3, [0, 1], q, a)
    print(D)
    E = env.transition_probabilities(9, 4, [0, 1], q, a)
    print(E)
    F = env.transition_probabilities(9, 5, [0, 1], q, a)
    print(F)
    G = env.transition_probabilities(9, 6, [0, 1], q, a)
    print(G)
    H = env.transition_probabilities(9, 7, [0, 1], q, a)
    print(H)
    I = env.transition_probabilities(9, 8, [0, 1], q, a)
    print(I)
    J = env.transition_probabilities(9, 9, [0, 1], q, a)
    print(J)
    K = env.transition_probabilities(9, 10, [0, 1], q, a)
    print(K)

    print('Here', A + B + C + D + E + F + G + H + I + J + K)  # should be equal to 1
    '''


    
    rl_utils = RL_Utils(env)
    ############ BENCHMARKING ############
    model_based_algorithms = ModelBased(env)


    # *** (1) and (2) are model based methods *** #
    # (1) ---------- Execute the value iteration algorithm -------------- #
    
    
    startVI = time.time()
    V, pi, Q, value_evolution = model_based_algorithms.value_iteration(gamma=1.0, epsilon=1e-10)
    endVI = time.time()
    # Calculate states and iterations
    states = np.arange(len(env.state_space))
    iterations = np.arange(value_evolution.shape[0])
    # Plot value evolution
    rl_utils.plot_value_evolution(value_evolution, states, iterations, 'Value Iteration')
    rl_utils.plot_q_values(Q, 'Value Iteration')
    print("Execution time for Value Iteration: ", endVI - startVI, "seconds \n")

    print('\n--- Value function for value iteration ---- \n')
    print(V)

    print('\n--- Q values for value iteration ---- \n')
    print(Q)

    print('\n--- Policy after value iteration ---- \n')
    for s in range(len(env.state_space) - 1):
        batch = env.action_space[pi(s)]
        print('\n If user watches content s= ', s, 'we recommend ', batch)
        print('\n Relevance between s and batch[0]: ', U[s, batch[0]])
        print('\n Relevance between s and batch[1]: ', U[s, batch[1]])
        print('\n Cache cost between s and batch[0]', env.cached_matrix[s, batch[0]])
        print('\n Cache cost between s and batch[1]', env.cached_matrix[s, batch[1]])
        print('\n-------------------------------------------\n')

    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')

    # (2) ---------- Execute the policy iteration algorithm -------------- #
    trans_prob_array = env.transition_probability_matrix()
    startPI = time.time()
    V2, pi2, Q2, valEvol = model_based_algorithms.policy_iteration(gamma=1.0, epsilon=1e-10)
    states = np.arange(len(env.state_space))
    iterations = np.arange(value_evolution.shape[0])
    rl_utils.plot_value_evolution(value_evolution, states, iterations, 'Policy Iteration')

    endPI = time.time()
    print("Execution time for Policy Iteration: ", endPI - startPI, "seconds \n")
    rl_utils.plot_q_values(Q2, 'Policy Iteration')
    print('\n--- Policy after policy iteration ---- \n')
    for s in range(len(env.state_space) - 1):
        batch = env.action_space[pi2(s)]
        print('\nIf user watches content s=', s, 'we recommend ', batch)
        print('\nRelevance between s and batch[0]: ', U[s, batch[0]])
        print('\nRelevance between s and batch[1]: ', U[s, batch[1]])
        print('\nCache cost between s and batch[0]', env.cached_matrix[s, batch[0]])
        print('\nCache cost between s and batch[1]', env.cached_matrix[s, batch[1]])
        print('\n----------------------------------------------------\n')

    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')


    pi_PolicyIteration = pi2

    # *** (3) and (4) are model free methods *** #
    model_free_algorithms = ModelFree(env)

    # (3) ---------- Execute SARSA algorithm -------------- #
    startSARSA = time.time()
    episodes_S = 100000
    Q_SARSA, pi_SARSA = model_free_algorithms.SARSA(num_episodes=episodes_S, learning_rate=0.1, discount_factor=gamma)
    endSARSA = time.time()
    print("Execution time for SARSA: ", endSARSA - startSARSA, "seconds \n")

    print('\n--- Policy after SARSA ---- \n')
    for s in range(len(env.state_space) - 1):
        batch = env.action_space[pi_SARSA(s)]
        print('\nIf user watches content s=', s, 'we recommend ', batch)
        print('\nRelevance between s and batch[0]: ', U[s, batch[0]])
        print('\nRelevance between s and batch[1]: ', U[s, batch[1]])
        print('\nCache cost between s and batch[0]', env.cached_matrix[s, batch[0]])
        print('\nCache cost between s and batch[1]', env.cached_matrix[s, batch[1]])
        print('\n----------------------------------------------------\n')

    rl_utils.plot_q_values(Q_SARSA, 'SARSA')

    print('\n---************************************** ---- \n')
    if all(pi_SARSA(s) == pi2(s) for s in range(len(env.state_space) - 1)):
        print('The policies are the same')



    # (4) ---------- Execute the Q-learning algorithm -------------- #

    episodesQ = 60000 # number of episodes
    init_learning_rate = 0.01 # initial learning rate
    startLearningRate = time.time()
    learning_rate_schedule = model_free_algorithms.meta_train(episodesQ, init_learning_rate, gamma)
    endLearningRate = time.time()

    print("Execution time for learning rate schedule: ", endLearningRate - startLearningRate, "seconds \n")

    startQL = time.time()

    QQ, piQ, cost = model_free_algorithms.Q_learning_scheduled(episodesQ, learning_rate_schedule, gamma)

    print(learning_rate_schedule)

    endQL = time.time()
    print("Execution time for Q Learning: ", endQL - startQL, "seconds \n")

    print('\n--- Policy after Q learning ---- \n')
    for s in range(len(env.state_space) - 1):
        batch = env.action_space[piQ(s)]
        print('\nIf user watches content s=', s, 'we recommend ', batch)
        print('\nRelevance between s and batch[0]: ', U[s, batch[0]])
        print('\nRelevance between s and batch[1]: ', U[s, batch[1]])
        print('\nCache cost between s and batch[0]', env.cached_matrix[s, batch[0]])
        print('\nCache cost between s and batch[1]', env.cached_matrix[s, batch[1]])
        print('\n----------------------------------------------------\n')

    rl_utils.plot_q_values(QQ, 'Q Learning')

    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')
    print('\n---************************************** ---- \n')

    if all(piQ(s) == pi2(s) for s in range(len(env.state_space) - 1)):
        print('The policies are the same with  Policy Iteration')