"""
The functions in the script Q-learning.
"""

import numpy as np
from functools import partial


def is_final(P_dict_state):
    """
    Args:
        P_dict[x]

    Returns:
        the result of check whether the state is final.
    """
    return len(P_dict_state[0]) == 1 and P_dict_state[0][0][3]


def get_random_Q(env):
    """
    Initialize a random Q-table:
    nS x nA {rq: rename vars in my file! }
    """
    nS, nA = env.observation_space.n, env.action_space.n

    P_dict = env.env.P
    final_states = np.where([is_final(P_dict[x]) for x in P_dict.keys()])[0]

    Q = np.random.random(size=(nS, nA))
    Q[final_states, :] = 0.0
    return Q


def compute_policy_by_Q(Q):
    """
    Determine the optimal deterministic policy
    for the given Q-table.

    Args:
        Q (ndarray of shape (nS, nA)): Q-table

    Returns:
        array of shape (nS,): the indices of actions of the optimal policy.
    """
    return np.argmax(Q, axis=1)


def epsilon_greedy_action(greedy, nA, epsilon):
    """
    Select the epsilon-greedy action.

    Args:
        greedy: the best action according to the current Q-table.
        nA: total number of actions
        epsilon: the probability of choosing an action at random.

    Returns:
        the index of the chosen action (int).
    """
    return greedy if np.random.rand() > epsilon else np.random.randint(nA)


def Q_learning_episode(env, pi, Q, alpha=0.05, epsilon=0.0, gamma=0.9):
    """
    Implements the algorithm OF Q_learning at:
    https://habrastorage.org/webt/wf/6x/fi/wf6xfiyazgu0echvfsw8d9-oly4.png

    Args:
        env : the environment.
        pi (array of size (nS,)): policy.
        Q  (matrix of size (nS, nA)): the Q-table.
        alpha: the learning rate. Defaults to 0.05
        epsilon: the probability of random action.  Defaults to 0.0
        gamma: the discount rate.  Defaults to 0.9

    Returns:
        This function updates the matrix Q.
    """

    nA = env.action_space.n

    env.reset()
    s = env.env.s
    a = epsilon_greedy_action(pi[env.env.s], nA, epsilon)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(a)
        s_prime = observation
        a_prime = epsilon_greedy_action(pi[observation], nA, epsilon)

        Q[s][a] = (Q[s][a]
                   + alpha * (reward + gamma * np.max(Q[s_prime]) - Q[s][a]))
        s, a = s_prime, a_prime
        if terminated:
            break


def Q_learning(environment, alpha=0.1, epsilon=0.1, gamma=0.9, total_episodes=10000):
    """
    The Q-learning algorithm for approaching the optimal policy.

    Args:
        environment
            (gymnasium.wrappers.order_enforcing.OrderEnforcing):
            the environment.
        discount_factor (float between 0 and 1, optional):
            the discounting factor, which attenuates future rewards.
            Defaults to 1.0.
        theta (float>0, optional):
            the threshold of change, below which the iterations stop.
            Defaults to 1e-9.
        max_iterations (int, optional): the maximum number of steps.
            Defaults to 10.

    Returns:
        the policy : function state -> action.
    """
    Q = get_random_Q(environment)
    pi = compute_policy_by_Q(Q)

    # for n in tqdm(range(total_episodes)):
    for n in range(total_episodes):
        Q_learning_episode(environment, pi, Q, alpha=alpha, epsilon=epsilon, gamma=gamma)
        pi = compute_policy_by_Q(Q)

    # convert the policy to function state -> action
    policy = partial(np.take, pi)
    return policy, pi
