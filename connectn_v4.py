#!/usr/bin/env python
import numpy as np
import random
import itertools as it
import os
import copy

import gym
from gym import spaces

class ConnectNEnv(gym.Env):
    """
    Methods this class has to contain:

    def reset() --> state
    def step(action) --> state, reward, done, info
    def render() --> # Render board

    """
    metadata = {'render.modes': ['human']}

    VISUALISATION = {
        1.: '\x1b[0;31;40m' + 'X' + '\x1b[0m',  # Color red
        -1.: '\x1b[0;32;40m' + 'O' + '\x1b[0m',  # Color green
        0.: ' '
    }

    def __init__(self, game_mode=['learner', 'random'], agent=None):
        # Game mode must be lenght of tow
        assert len(game_mode) == 2
        # Only specific game modes are allowed
        for gm in game_mode:
            assert gm in ['learner', 'random', 'human', 'agent']

        self.game_mode = game_mode
        
        self.__version__ = "0.1.0"

        self.n = 3  # Required disks to win
        self.m = 4  # Size of field

        self.curr_step = -1
                
        self.agent = agent

        self.name = f'Connect{self.n}'

        self.action_space = spaces.Discrete(self.m)
        self.observation_space = spaces.Box(low=-1, high=1, dtype=np.int, shape=(self.m, self.m))
        #self.observation_space = spaces.Discrete(self.m*self.m)

        self.curr_episode = -1
        self.action_episode_memory = []

        # Manipulate rewards
        self.reward_win = 1
        self.reward_lost = -10
        self.reward_set_on_full_column = -0.5
        self.reward_draw = 0

        self.create_winning_templates()
        self.reset()

    def _step_learner(self, action):
        # Check for valid turn. If not, add a small negative reward
        if action not in self.possible_turns:
            return self.board, self.reward_set_on_full_column, True, {}

        # Place disk
        self._place_disk(action)
        
        # Check for winning
        if self.check_for_winning():
            return self.board, self.reward_win, True, {}

        # Board is full --> no winner
        elif self.disks_set >= self.m * self.m:
            return self.board, self.reward_draw, True, {}

        # Game goes on, toggle the disk
        self.toggle_disk()

        # Default return
        return self.board, 0, False, {}

    def _step_agent(self):
        # Evaluate action from agent
        self.agent.training = False
        action = self.agent.forward(self.board)
        self.agent.training = True

        # If turn is invalid, choose random turn
        if action not in self.possible_turns:
            action = self._get_random_action()

        # Place disk and evaluate
        self._evaluate_opponent_turn(action)

    def _step_random(self):
        # Choose random (possible) action
        action = self._get_random_action()

        # Place disk and evaluate
        self._evaluate_opponent_turn(action)

    def _step_human(self, action):
        # TODO: Write human interface here
        return

    def _evaluate_opponent_turn(self, action):
        # Place disk
        self._place_disk(action)

        # Check for winning
        if self.check_for_winning():
            return self.board, self.reward_lost, True, {}

        # Board is full --> no winner
        elif self.disks_set >= self.m * self.m:
            return self.board, self.reward_draw, True, {}

        # Game goes on, toggle the disk
        self.toggle_disk()

        # Default return
        return self.board, 0, False, {}

    def _place_disk(self, action):
        # Place disk
        disks_in_column = np.sum(np.abs(self.board[:, action]))
        self.board[int(self.m - disks_in_column - 1), action] = \
            self.current_disk

        # Adjust turn options, of column is full
        if disks_in_column + 1 == self.m:
            self.possible_turns.remove(action)

        # Count up
        self.disks_set += 1

    def step(self, action):
        result = self._step_learner(action)
        if result is not None:
            return result

        if 'random' in game_mode:
            result = self._step_random()
            if result is not None:
                return result


    
    def single_step(self, action):
        # Check for valid turn        
        if action not in self.possible_turns:
            return self.board, -0.5, True, {}

        # Turn is valid
        self.disks_set += 1

        # Place disk
        disks_in_column = np.sum(np.abs(self.board[:, action]))
        self.board[int(self.m - disks_in_column - 1), action] = \
            self.current_disk

        # Adjust turn options, of column is full
        if disks_in_column + 1 == self.m:
            self.possible_turns.remove(action)

        # Check if learner won the game
        if self.check_for_winning():
            return self.board, 1, True, {}

        # Board is full --> no winner
        elif self.disks_set >= self.m * self.m:
            return self.board, 0, True, {}
        
        self.toggle_disk()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        # Update statistics
        self.curr_episode += 1
        self.action_episode_memory.append([])

        self.possible_turns = set(range(self.m))
        self.board = self.create_empty_board()

        self.disks = it.cycle([1, -1])
        self.toggle_disk()

        self.disks_set = 0
        self.game_over = False

        self.winner = None

        self.reward = 0
        self.terminal = False

        return self.board

    def seed(self, seed):
        random.seed(seed)
        np.random.seed

    def observe(self):
        return self.board, self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def toggle_disk(self):
        self.current_disk = next(self.disks)

    def create_empty_board(self):
        return np.zeros((self.m, self.m))

    def create_winning_templates(self):
        # Horizontal and vertical
        winning_templates = []
        for row in range(self.m):
            for col in range(self.m):

                # Horizontal lines
                if col + self.n <= self.m:
                    board_hor = self.create_empty_board()
                    board_hor[row, col:int(col+self.n)] = 1
                    winning_templates.append(board_hor)

                # Vertical lines
                if row + self.n <= self.m:
                    board_ver = self.create_empty_board()
                    board_ver[row:int(row+self.n), col] = 1
                    winning_templates.append(board_ver)

                # Diagonal lines (up and down)
                if (col + self.n <= self.m) & (row + self.n <= self.m):
                    board_diag_down = self.create_empty_board()
                    board_diag_down[row:, col:][np.diag_indices(self.n)] = 1
                    board_diag_up = np.fliplr(board_diag_down)
                    winning_templates.append(board_diag_down)
                    winning_templates.append(board_diag_up)

        self.winning_templates = np.array(winning_templates)

    def check_for_winning(self):
        if np.max(
            np.abs(
                np.sum(
                    np.sum(self.board * self.winning_templates, axis=1),
                    axis=1
                )
            )
        ) == self.n:
            return True
        return False

    def set_disk(self, column):
        # Check for valid turn
        if column not in self.possible_turns:
            print(
                f'Not possible, player {self.current_disk}. Please choose one of the following '
                f'columns: {self.possible_turns}. Your choice: {column}'
            )
            return False

        # Place disk
        disks_in_column = np.sum(np.abs(self.board[:, column]))
        self.board[int(self.m - disks_in_column - 1), column] = \
            self.current_disk

        # Adjust turn options, of column is full
        if disks_in_column + 1 == self.m:
            self.possible_turns.remove(column)

        return True
    
    def display_board(self):
        for row in self.board:
            text = '|' + ''.join([self.VISUALISATION[val] for val in row]) + '|'
            print(text)
        # Make it more human readable
        print(f' {"".join([str(i) for i in range(self.m)])} ')

    def _get_random_action(self):
        self.set_disk(np.random.choice(list(self.possible_turns)))
        
    def render(self, mode='human', close=False):
        self.display_board()
        return

    def stack_on_top(self):
        # Try to find a good turn
        for row in range(self.m - 1, -1, -1):
            row_ = self.board[row]
            for i, col in enumerate(row_):
                if col != 0 and i in self.possible_turns:
                    self.set_disk(int(i))
                    return

        # Otherwise, apply random turn
        self._get_random_action()