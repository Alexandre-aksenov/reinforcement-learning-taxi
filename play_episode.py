"""
The functions for playing episodes.
"""

import matplotlib.pyplot as plt
from IPython import display

import time


def show_state(fig, ax, environment):
    ax.imshow(environment.render())
    time.sleep(0.5)
    display.display(fig)
    display.clear_output(wait=True)


def play_one_move(environment, state, policy, graph_objects):
    """
    Play one move.

    Args:
        environment (gymnasium.wrappers.time_limit.TimeLimit): the environment.
        state (int): the encoding of the state before the action.
        policy (function state -> action):
            the policy to test.
        graph_objects (tuple or None):
            (figures, axes) for plotting,
            or skip the plotting operations if None.

    Returns:
        next_state (int): the encoding of the state after the action.
        reward (int):  the reward for the action.
        terminated (bool):
            indicates whether this action brings the episode to a final state.
        truncated (bool): indicates whether the episode is over by time.
    """

    # Select best action to perform in the current state
    action = policy(state)

    # Perform an action and observe how environment acted in response
    next_state, reward, terminated, truncated, __ = environment.step(action)

    if (graph_objects):
        fig_board, ax_board = graph_objects
        ax_board.clear()
        show_state(fig_board, ax_board, environment)

    return next_state, reward, terminated, truncated


def play_one_episode(environment, policy, show=False):
    """
    Play one episode.

    Args:
        environment (gymnasium.wrappers.time_limit.TimeLimit): the environment.
        policy (function state -> action):
            the policy to test.
        show (bool, optional): option to plot the episode, Defaults to False.

    Returns:
        win (bool): the outcome of the episode.
        total_reward (int): the reward of the episode.
        lst_states (list of ints): the list of states in the episode.
        graph_objects: figure, axis if show=True ; None otherwize.
    """

    if (show):
        fig_board, ax_board = plt.subplots(figsize=(4, 3))
        display.clear_output(wait=True)

    graph_objects = (fig_board, ax_board) if show else None

    terminated = False
    truncated = False
    total_reward = 0

    state = environment.reset()[0]  # the code of the initial state (int)
    lst_states = [state]

    if (show):
        show_state(fig_board, ax_board, environment)

    while not terminated and not truncated:
        # play one move.

        next_state, reward, terminated, truncated = play_one_move(
            environment, state, policy, graph_objects)
        lst_states.append(next_state)
        # Summarize total reward
        total_reward += reward
        # Update current state
        state = next_state

    # Results of the episode.
    # According to the description of the environment,
    # getting a positive final reward is equivalent to victory.
    win = (terminated and reward > 0)

    if (show):
        if terminated:
            print("Well done!")
        if truncated:
            print("Time limit exceeded. Try again.")

    return win, total_reward, lst_states, graph_objects


def play_episodes(environment, n_episodes, policy):
    """
    Validation.
    Play 'n_episodes' episodes with the same policy.

    Args:
        environment (gymnasium.wrappers.time_limit.TimeLimit): the environment
        n_episodes (int): the number of episode to play.
        policy (function state -> action):
            the policy to test.

    Returns:
        wins: the number of wins.
        total_reward: the reward cumulated over all episodes.
        average_reward: the average reward over all episodes.
        lst_lst_states : the list of states. New output.
    """
    wins = 0
    total_reward = 0
    lst_lst_states = []

    # one episode
    for __ in range(n_episodes):
        win, reward, lst_states, __ = play_one_episode(environment, policy)
        wins += win
        total_reward += reward
        lst_lst_states.append(lst_states)

    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward, lst_lst_states
