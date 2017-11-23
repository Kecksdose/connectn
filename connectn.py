#!/usr/bin/env python
import argparse
import numpy as np
import itertools as it
import os

# For statistics
try:
    from tqdm import trange
except:
    trange = range
from collections import Counter

parser = argparse.ArgumentParser(
    description='Fully implemented Connect 4 (N) game.'
)
parser.add_argument(
    '-n',
    help='Required disks to win.',
    default=4,
    type=int
)
parser.add_argument(
    '-m',
    help='Dimensions of the board.',
    default=7,
    type=int
)
parser.add_argument(
    '-M',
    '--mode',
    help='Game mode.',
    default='2c',
    type=str,
    choices=['2h', '1h1c', '2c']
)
args = parser.parse_args()


class ConnectN():

    VISUALISATION = {
        1.: '\x1b[0;31;40m' + 'X' + '\x1b[0m',  # Color red
        -1.: '\x1b[0;32;40m' + 'O' + '\x1b[0m',
        0.: ' '
    }

    def __init__(self, n, m, mode):
        self.n = n  # Size of field
        self.m = m  # Required disks to win
        self.mode = mode  # Game mode
        self.create_winning_templates()

        self.refresh_game()

    def refresh_game(self):
        self.possible_turns = set(range(self.m))
        self.board = self.create_empty_board()

        self.disks = it.cycle([1, -1])
        self.toggle_disk()

        self.disks_set = 0
        self.game_over = False

        self.winner = None

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
                'Not possible. Please choose one of the following '
                'columns: {self.possible_turns}.'
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

    def cpu_turn(self):
        # TODO: Machine learning
        self.set_disk(np.random.choice(list(self.possible_turns)))

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

if __name__=='__main__':
    # TODO: Catch more dump inputs.
    if args.n > args.m:
        raise ValueError(
            f'n should be lower than m, or at least equal. ({args.n} vs {args.m})'
        )
    elif args.n < 2:
        raise ValueError(
            f'n should be greater than 1. ({args.n})'
        )

    connectn = ConnectN(args.n, args.m, args.mode)

    if 'h' in args.mode:
        connectn.play()
    else:
        winners = []
        for i in trange(1000):
            connectn.play()
            winners.append(connectn.winner)
            connectn.refresh_game()
        print(Counter(winners))
