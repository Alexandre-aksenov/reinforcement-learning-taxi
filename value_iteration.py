"""
Implementation of the Value Iteration algorithm.

Input: the environment.
Output: the policy which is learned by the algorithm
(function state -> action).
"""

import numpy as np
from functools import partial


def one_step_lookahead(environment, state, V, discount_factor):
    """
    The values of all actions in the given state and the given environment,
    according to the array 'V' of previous values of all states.

    Args:
        environment (gymnasium.wrappers.order_enforcing.OrderEnforcing):
            the environment.
        state (int): the index of state, for which the new value is sought.
        V (_type_): the array of previous values of all states.
        discount_factor (float between 0 and 1, optional):
            the discounting factor, which attenuates future rewards.

    Implements the formula:
    $$Q_i(s, a) = sum_{s'} P(s' | s,a) cdot [ r(s,a,s') + gamma V_{i}(s')]$$

    Returns:
        ndarray of size nA: The new action values.
    """
    action_values = np.zeros(environment.action_space.n)
    for action in range(environment.action_space.n):  # loop over actions
        for probability, next_state, reward, _ in environment.P[state][action]:
            # environment.P : double dict.
            # Loop over the possible outcomes of the action.
            outcome_value = (probability
                            * (reward + discount_factor * V[next_state]))
            action_values[action] += outcome_value
    return action_values


def V_to_policy(environment, V, discount_factor):
    """
    Create a deterministic policy using the optimal value function.

    Args:
        environment
            (gymnasium.wrappers.order_enforcing.OrderEnforcing):
            the environment.
        V (np.ndarray of shape nS x nA): the values of actions.
        discount_factor (float between 0 and 1, optional):
            the discounting factor, which attenuates future rewards.

    Returns:
        pi (array of unsigned ints of shape (nS,)):
            the indices of actions in optimal policy
    """
    nS = environment.observation_space.n

    pi = np.zeros(nS, dtype=np.uint)
    for state in range(nS):
        # One step lookahead to find the best action for this state
        action_value = one_step_lookahead(
            environment, state, V, discount_factor)
        # Select best action based on the highest state-action value
        best_action = np.argmax(action_value)
        # Update the policy to perform a better action at a current state
        pi[state] = best_action
    return pi


def mat_policy_val_iter(
        environment, discount_factor=1.0, theta=1e-9, max_iterations=10):
    """
    The value iteration algorithm for approaching the optimal policy.

    Implements the formula:
    $$V_{(i+1)}(s) = max_a Q_i(s,a)$$

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
        pi (array of unsigned ints of shape (nS,)):
            the indices of actions in optimal policy;

        V (np.ndarray of shape nS x nA): the values of actions.
    """

    nS = environment.observation_space.n
    # Initialize the matrix of values (array indexed by codes of states).
    V = np.zeros(nS)
    for i in range(int(max_iterations)):
        # The biggest amount of change of value among states
        delta = 0
        # Update each state
        for state in range(nS):
            # Local variables to this loop: action_value (),
            # best_action_value: the value of the best action,
            # the new value of the state.
            # Updated variables: delta , V.

            # One-step lookahead to calculate state-action values
            # The values of actions for this state.
            action_value = one_step_lookahead(
                environment, state, V, discount_factor)

            # added printouts (to remove for debugging).
            # print('action_value is a:', type(action_value)) # ndarray
            # print('of shape', action_value.shape)  # (6,)

            # Select best action to perform
            # based on the highest state-action value
            best_action_value = np.max(action_value)
            # Compute change in value , update 'delta'
            delta = max(delta, np.abs(V[state] - best_action_value))
            # Update the value function for current state
            V[state] = best_action_value

        # Check if we can stop
        if delta < theta:
            print(f'Value-iteration converged at iteration#{i}.')
            break

    # Define a deterministic policy
    pi = V_to_policy(environment, V, discount_factor)
    return pi, V


def value_iteration(
        environment, discount_factor=1.0, theta=1e-9, max_iterations=10):
    """
    The value iteration algorithm for approaching the optimal policy.

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

    pi, __ = mat_policy_val_iter(
        environment.env, discount_factor, theta, max_iterations)

    policy = partial(np.take, pi)
    return policy, pi
