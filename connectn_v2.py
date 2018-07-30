#!/usr/bin/env python
import numpy as np
import itertools as it
import os


class ConnectN():

    VISUALISATION = {
        1.: '\x1b[0;31;40m' + 'X' + '\x1b[0m',  # Color red
        -1.: '\x1b[0;32;40m' + 'O' + '\x1b[0m',  # Color green
        0.: ' '
    }

    def __init__(self, n, m, mode):
        self.name = f'Connect{n}'
        self.n = n  # Required disks to win
        self.m = m  # Size of field
        self.mode = mode  # Game mode
        self.enable_actions = list(range(m))
        self.create_winning_templates()

        self.reset()

    def reset(self):
        self.possible_turns = set(range(self.m))
        self.board = self.create_empty_board()

        self.disks = it.cycle([1, -1])
        self.toggle_disk()

        self.disks_set = 0
        self.game_over = False

        self.winner = None

        self.reward = 0
        self.terminal = False

    def observe(self):
        return self.board, self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def update(self, action):
        # Check for valid turn
        if action not in self.possible_turns:
            return

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
            self.reward = self.current_disk
            self.terminal = True
            return

        # Board is full --> no winner
        elif self.disks_set >= self.m * self.m:
            self.reward = 0
            self.game_over = True
            return

        # Second player: 25% Random turn, 75% Stack on top.
        self.toggle_disk()
        if np.random.random() > 0.75:
            self.random_turn()
        else:
            self.stack_on_top()
        self.disks_set += 1

        # Again: Check if this player won the game
        if self.check_for_winning():
            self.reward = self.current_disk
            self.terminal = True
            return

        # Again: Check if board is full
        elif self.disks_set >= self.m * self.m:
            self.reward = 0
            self.game_over = True
            return

        # Continue playing
        self.toggle_disk()


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

    def display_board(self):
        for row in self.board:
            text = '|' + ''.join([self.VISUALISATION[val] for val in row]) + '|'
            print(text)
        # Make it more human readable
        print(f' {"".join([str(i) for i in range(self.m)])} ')

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

    def human_turn(self):
        decision = None
        player = self.VISUALISATION[self.current_disk]
        while decision not in self.possible_turns:
            decision = input(
                f'Player {player}, please choose a'
                f' column ({self.possible_turns}): '
            )
            try:
                decision = int(decision)
            except:
                pass

        self.set_disk(decision)

    def random_turn(self):
        # TODO: Machine learning
        self.set_disk(np.random.choice(list(self.possible_turns)))

    def stack_on_top(self):
        # Try to find a good turn
        for row in range(self.m - 1, -1, -1):
            row_ = self.board[row]
            for i, col in enumerate(row_):
                if col != 0 and i in self.possible_turns:
                    self.set_disk(int(i))
                    return

        # Otherwise, apply random turn
        self.random_turn()

    def refresh_screen(self):
        os.system('clear')

    def play(self):
        # TODO: Tidy up game modes
        if 'h' in self.mode:
            self.refresh_screen()
            self.display_board()
        while not self.game_over:
            ### Play against CPU
            if self.mode == '1h1c':
                if self.current_disk == 1:
                    self.human_turn()
                    self.refresh_screen()
                else:
                    self.cpu_turn()
                    self.display_board()
            ### Play human vs human
            elif self.mode == '2h':
                self.human_turn()
                self.refresh_screen()
                self.display_board()
            ### Play CPU vs CPU
            else:
                self.cpu_turn()

            self.disks_set += 1

            if self.check_for_winning():
                self.game_over = True
                self.winner = self.current_disk
                if 'h' in self.mode:
                    self.refresh_screen()
                    self.display_board()
                    print(
                        f'Winner: {self.VISUALISATION[self.current_disk]}'
                    )

            elif self.disks_set >= self.m * self.m:
                self.game_over = True
                self.winner = 0
                if 'h' in self.mode:
                    self.refresh_screen()
                    self.display_board()
                    print(f'No Winner.')

            self.toggle_disk()
