#!/usr/bin/python3
import sys

import argparse
import curses
from curses import wrapper
import random

EMPTY = 0
PAWN = 1
ROOK = 2
KNIGHT = 3
BISHOP = 4
KING = 5
QUEEN = 6

WHITE = 0x10
BLACK = 0x8

PIECES = {EMPTY: ' ',
          WHITE | PAWN: '\u2659', WHITE | ROOK: '\u2656', WHITE | KNIGHT: '\u2658',
          WHITE | BISHOP: '\u2657', WHITE | KING: '\u2654', WHITE | QUEEN: '\u2655',
          BLACK | PAWN: '\u265F', BLACK | ROOK: '\u265C', BLACK | KNIGHT: '\u265E',
          BLACK | BISHOP: '\u265D', BLACK | KING: '\u265A', BLACK | QUEEN: '\u265B'}

VALUE = {EMPTY: 0, PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0}

NAMES = {EMPTY: '', PAWN: 'pawn', ROOK: 'rook', KNIGHT: 'knight', BISHOP: 'bishop', KING: 'king', QUEEN: 'queen',
         WHITE: 'white', BLACK: 'black'}

BOX = {'HORIZONTAL': '\u2501' * 3, 'VERTICAL': '\u2503',
       'TOP T': '\u2533', 'BOTTOM T': '\u253B', 'LEFT T': '\u2523', 'RIGHT T': '\u252B',
       'TOP LEFT': '\u250F', 'TOP RIGHT': '\u2513', 'BOTTOM RIGHT': '\u251B', 'BOTTOM LEFT': '\u2517',
       'CROSS': '\u254B'}

TEXT_COLOR = curses.COLOR_BLACK
# Add 1 to get dark colors
BOARD_COLOR = 1
YELLOW = 3
GREEN = 5
RED = 7
CYAN = 9
BLUE = 11
MAGENTA = 13


class ChessState:

    def __init__(self, to_move=WHITE, board=None, captured=None):
        # Indicates which players turn it is
        self.to_move = to_move
        # Indicates of the given color captured a piece in their last turn
        self.captured_last_move = {WHITE: False, BLACK: False}
        self.rook_moved = {BLACK: [False, False], WHITE: [False, False]}
        self.king_moved = {BLACK: False, WHITE: False}
        self.capture_moves = {WHITE: None, BLACK: None}         # type: dict
        self.non_capture_moves = {WHITE: None, BLACK: None}     # type: dict

        if board is None:
            self.captured = []
            self.board = [[0] * 8 for _ in range(8)]
            self.board[0][0] = self.board[0][7] = BLACK | ROOK
            self.board[0][1] = self.board[0][6] = BLACK | KNIGHT
            self.board[0][2] = self.board[0][5] = BLACK | BISHOP
            self.board[0][3] = BLACK | QUEEN
            self.board[0][4] = BLACK | KING
            self.board[1] = [BLACK | PAWN] * 8

            self.board[6] = [WHITE | PAWN] * 8
            self.board[7][0] = self.board[7][7] = WHITE | ROOK
            self.board[7][1] = self.board[7][6] = WHITE | KNIGHT
            self.board[7][2] = self.board[7][5] = WHITE | BISHOP
            self.board[7][3] = WHITE | QUEEN
            self.board[7][4] = WHITE | KING
        else:
            self.captured = captured
            self.board = board

    def value(self):
        """
        Gives the value of the current state. The value is the sum of all the values of the white pieces minus the sum
        of the values of the black pieces. The white player is aiming to increase this value, while the black player is
        aiming to decrease it. The pieces have the following values
            Pawn:   1
            Knight: 3
            Bishop: 3
            Rook:   5
            Queen:  9
        The king is not given a value, since it cannot be taken.
        There are a few special value states:
            - Black checkmate is worth 1e10 and white checkmate is worth -1e10 (i.e. nothing is better than checkmate)
            - Stalemate is worth zero points
        Returns:
            The value of the current state
        """
        if self.is_checkmate():
            return 1000 if self.to_move is BLACK else -1000
        if self.is_stalemate():
            return 0

        value = 0
        for row in range(8):
            for col in range(8):
                value += VALUE[self.get_piece((row, col))] * (1 if self.get_color((row, col)) is WHITE else -1)
        return value

    def get_color(self, pos):
        """
        Returns the color of the piece at the specified position

        Args:
            pos: position of the square to get the pieces color of. Tuple of the form (row, col)
        Return:
            int: WHITE if the piece at pos is white, BLACK if the piece at pos is black and EMPTY if it is empty
        """
        return self.board[pos[0]][pos[1]] & 0x18

    def get_piece(self, pos):
        """
        Returns the piece at the specified position

        Args:
            pos: position of the square to get the piece from. Tuple of the form (row, col)
        Return:
            int: PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING, or EMPTY
        """
        return self.board[pos[0]][pos[1]] & 0x7

    def copy(self):
        copy = ChessState(to_move=self.to_move, board=[row[:] for row in self.board], captured=self.captured[:])
        copy.captured_last_move = {BLACK: self.captured_last_move[BLACK], WHITE: self.captured_last_move[WHITE]}
        copy.king_moved = {BLACK: self.king_moved[BLACK], WHITE: self.king_moved[WHITE]}
        copy.rook_moved = {WHITE: [self.rook_moved[WHITE][0], self.rook_moved[WHITE][1]],
                           BLACK: [self.rook_moved[BLACK][0], self.rook_moved[BLACK][1]]}
        return copy

    def is_pawn_promotion_move(self, move):
        """
        Returns True if the given move results in a pawn promotion. This happens when a pawn reaches rank 7
        Args:
            move (Move): The move to determine if it is a pawn promotion move
        Returns:
            bool: True if the move results in pawn promotion, False otherwise
        """
        if self.get_piece(move.src) is PAWN:
            if self.rank(move.dst[0], self.get_color(move.src)) == 7:
                return True
        return False

    def draw_square(self, scr, pos, color):
        """
        Draws the square specified by the pos argument, in the given color. The checkered pattern of a chess board
        is handled automatically by this function (i.e. don't pass in different colors to create the checkered pattern,
        it is done automatically)

        Args:
            scr: The screen to draw the square on
            pos (tuple): The position of the square to draw in the form (row, col)
            color (int): The color to draw the square
        """
        scr.addstr(1 + pos[0], 2 + pos[1] * 3, f' {PIECES[self.board[pos[0]][pos[1]]]} ',
                   curses.color_pair(color + (pos[0] + pos[1]) % 2))

    def draw_captured_pieces(self, scr):
        """Draws all the pieces that have been captured to the side of the board"""
        scr.addstr(1, 27, ' ' * 11, curses.color_pair(BOARD_COLOR))
        scr.addstr(2, 27, ' ' * 11, curses.color_pair(BOARD_COLOR))
        scr.addstr(3, 27, ' ' * 11, curses.color_pair(BOARD_COLOR))

        scr.addstr(6, 27, ' ' * 11, curses.color_pair(BOARD_COLOR))
        scr.addstr(7, 27, ' ' * 11, curses.color_pair(BOARD_COLOR))
        scr.addstr(8, 27, ' ' * 11, curses.color_pair(BOARD_COLOR))

        num_white = 0
        num_black = 0
        last_black_index = -1
        last_white_index = -1
        for i in range(len(self.captured)):
            if self.captured[i] & 0x18 is WHITE:
                scr.addstr(6 + num_white // 5, 28 + 2 * (num_white % 5),
                           f'{PIECES[self.captured[i]]}', curses.color_pair(BOARD_COLOR))
                num_white += 1
                last_white_index = i
            else:
                scr.addstr(1 + num_black // 5, 28 + 2 * (num_black % 5),
                           f'{PIECES[self.captured[i]]}', curses.color_pair(BOARD_COLOR))
                num_black += 1
                last_black_index = i

        num_white -= 1
        num_black -= 1
        if self.captured_last_move[BLACK] and last_white_index >= 0:
            scr.addstr(6 + num_white // 5, 27 + 2 * (num_white % 5),
                       f' {PIECES[self.captured[last_white_index]]} ',
                       curses.color_pair(GREEN if self.to_move is WHITE else RED))

        if self.captured_last_move[WHITE] and last_black_index >= 0:
            scr.addstr(1 + num_black // 5, 27 + 2 * (num_black % 5),
                       f' {PIECES[self.captured[last_black_index]]} ',
                       curses.color_pair(GREEN if self.to_move is BLACK else RED))

    def draw_move(self, scr, move):
        """
        Draws a move. This will draw the src square as red, and the dst square as green
        Args:
            scr: The screen to draw the move onto
            move (Move): The move to draw
        """
        if move is None:
            return
        self.draw_square(scr, move.src, GREEN)
        self.draw_square(scr, move.dst, RED)

    def draw_possible_moves(self, scr, src, selected):
        """
        Draws the moves that can be made from a piece on a given square. If no moves can be made, this is indicated by
        drawing the square red. A selected square will also be drawn blue. If None is passed for this parameter, this
        will not be drawn

        Args:
            scr: The screen to draw the moves onto
            src (tuple): The square containing the piece to draw the moves for
            selected: The square to draw as selected. This is drawn blue
        """
        capture_moves, non_capture_moves = self.get_moves(src)
        moves = capture_moves + non_capture_moves
        # If the selection is the player and there are valid moves that the selected move can take, then
        # draw the selections green and draw the possible moves from that piece is cyan
        if self.get_color(src) is self.to_move and len(moves) > 0:
            self.draw_square(scr, src, GREEN)
            moves = [move.dst for move in moves]
            for move in moves:
                self.draw_square(scr, move, CYAN)

        # Otherwise, the piece on this square cannot be moved, so it is drawn red
        else:
            self.draw_square(scr, src, RED)

        if selected is not None:
            self.draw_square(scr, selected, BLUE)

    def draw_board(self, scr):
        # Draw each square
        for row in range(8):
            for col in range(8):
                self.draw_square(scr, (row, col), BOARD_COLOR)

    @staticmethod
    def rank(row, color):
        """
        Returns the rank of the given row from the perspective of the given player. The rank of a row is the number of
        rows from that colors back line. Returns an integer between 0 and 7
        Args:
            row (int): The row to determine the rank of
            color (int): The color to determine the rank with respect to
        Returns:
             int: The rank of the row from the given colors perspective (between 0 and 7)
        """
        if color is BLACK:
            return row
        return 7 - row

    def is_check(self, color=None):
        """
        Determines if the given color is currently in check
        Args:
            color (int): The color to determine if it is in check or not (defaults to the player whose turn it is)
        Returns:
            bool: True if the color is in check, False otherwise
        """
        if color is None:
            color = self.to_move
        capture_moves, _ = self.get_all_moves(WHITE if color is BLACK else BLACK, remove_check=False)
        for move in capture_moves:
            # An opponent move is able to capture the king if this is true
            if self.get_color(move.dst) is color and self.get_piece(move.dst) is KING:
                return True

        return False

    def is_checkmate(self, color=None):
        """
        Determines if the given color is currently in checkmate
        Args:
            color (int): The color to determine if it is in checkmate or not (defaults to the player whose turn it is)
        Returns:
            bool: True if the color is in checkmate, False otherwise
        """
        if color is None:
            color = self.to_move
        if self.is_check(color):
            capture_moves, non_capture_moves = self.get_all_moves(color)
            return len(capture_moves) + len(non_capture_moves) == 0
        return False

    def is_stalemate(self):
        """
        Determines if the game is in a stalemate
        Returns:
            bool: True if the game is in stalemate, False otherwise
        """
        if not self.is_check():
            capture_moves, non_capture_moves = self.get_all_moves()
            return len(capture_moves) + len(non_capture_moves) == 0
        return False

    def is_game_finished(self):
        """
        Determines if the game is finished in its current state. A game finishes when checkmate or stalemate has been
        reached
        Returns:
            bool: True if the game is finished, False otherwise
        """
        capture_moves, non_capture_moves = self.get_all_moves()
        return len(capture_moves) + len(non_capture_moves) == 0

    def move(self, move):
        """
        Performs the supplied move, and returns the state of the board after this move has been made.

        Args:
            move (Move): Move object describing the move to make
        Returns:
            ChessState: A chess state object containing the state after the move has been made
        """
        new_state = self.copy()
        # Check if a players rook is being moved for first time
        piece = self.get_piece(move.src)
        color = self.get_color(move.src)
        if piece is ROOK:
            if ChessState.rank(move.src[0], color) == 0:
                if move.src[1] == 0:
                    new_state.rook_moved[color][0] = True
                if move.src[1] == 7:
                    new_state.rook_moved[color][1] = True

        # Check if king moved for first time
        if piece is KING:
            new_state.king_moved[color] = True
            # Check if this is a castling move. If it is move the rook
            if move.src[1] - move.dst[1] == 2:
                new_state = new_state.move(Move((move.src[0], 0), (move.src[0], 3), color | ROOK))
            if move.src[1] - move.dst[1] == -2:
                new_state = new_state.move(Move((move.src[0], 7), (move.src[0], 5), color | ROOK))

        new_state.board[move.src[0]][move.src[1]] = EMPTY
        new_state.captured_last_move[color] = False
        if new_state.get_piece(move.dst) is not EMPTY:
            new_state.captured.append(new_state.board[move.dst[0]][move.dst[1]])
            new_state.captured_last_move[color] = True
        new_state.board[move.dst[0]][move.dst[1]] = move.final_piece
        new_state.to_move = WHITE if self.to_move is BLACK else BLACK
        return new_state

    def add_pawn_move(self, move, moves):
        """
        Adds a move made by a pawn to a list of moves. If the move results in a pawn promotion, adds all the possible
        promotions, otherwise just adds the move to the list
        Args:
            move: The move to add. Must be a move made by a pawn
            moves: The list of moves to add the pawn move to
        """
        color = self.get_color(move.src)
        if self.is_pawn_promotion_move(move):
            moves.append(Move(move.src, move.dst, color | ROOK))
            moves.append(Move(move.src, move.dst, color | BISHOP))
            moves.append(Move(move.src, move.dst, color | KNIGHT))
            moves.append(Move(move.src, move.dst, color | QUEEN))
        else:
            moves.append(move)

    def get_moves(self, pos, remove_check=True):
        """
        Returns a list of all the moves that a given piece can make

        Args:
            pos (tuple): the position of the piece to find moves for. Tuple of the form (row, col)
            remove_check (bool): whether moves that result in check should be removed. Ordinarily, these moves are
                illegal and so should be removed. However, when determining if the other player is in check, these
                moves must be considered. Thus, if getting moves to check if a player is in check, set remove_check to
                False so that these moves are not removed.

        Returns:
            list<Move>: All valid moves that the given piece can make
        """
        capture_moves = []
        non_capture_moves = []
        (row, col) = pos
        piece = self.get_piece(pos)
        color = self.get_color(pos)
        opponent = BLACK if color is WHITE else WHITE
        if piece is PAWN:
            direction = -1 if color is WHITE else 1
            # Diagonal capture moves
            if col != 0 and self.get_color((row + direction, col - 1)) is opponent:
                self.add_pawn_move(Move((row, col), (row + direction, col - 1), color | piece), capture_moves)

            if col != 7 and self.get_color((row + direction, col + 1)) is opponent:
                self.add_pawn_move(Move((row, col), (row + direction, col + 1), color | piece), capture_moves)

            # Forward, non capture moves
            if self.board[row + direction][col] is EMPTY:
                self.add_pawn_move(Move((row, col), (row + direction, col), color | piece), non_capture_moves)
                # The second space is only available if the first was empty
                if ChessState.rank(row, color) == 1 and self.board[row + 2 * direction][col] is EMPTY:
                    self.add_pawn_move(Move((row, col), (row + 2 * direction, col), color | piece), non_capture_moves)

        if piece is KNIGHT:
            for i in range(4):
                dst_row = row + (2 if i % 2 == 0 else -2)
                dst_col = col + (1 if i // 2 == 0 else -1)
                if 0 <= dst_col < 8 and 0 <= dst_row < 8:
                    dst_color = self.get_color((dst_row, dst_col))
                    if dst_color is opponent:
                        capture_moves.append(Move((row, col), (dst_row, dst_col), color | piece))
                    elif dst_color is EMPTY:
                        non_capture_moves.append(Move((row, col), (dst_row, dst_col), color | piece))

                dst_row = row + (1 if i % 2 == 0 else -1)
                dst_col = col + (2 if i // 2 == 0 else -2)
                if 0 <= dst_col < 8 and 0 <= dst_row < 8:
                    dst_color = self.get_color((dst_row, dst_col))
                    if dst_color is opponent:
                        capture_moves.append(Move((row, col), (dst_row, dst_col), color | piece))
                    elif dst_color is EMPTY:
                        non_capture_moves.append(Move((row, col), (dst_row, dst_col), color | piece))

        if piece is BISHOP or piece is QUEEN:
            directions = [[(row + i, col + i) for i in range(1, min(8 - row, 8 - col))],
                          [(row + i, col - i) for i in range(1, min(8 - row, col + 1))],
                          [(row - i, col + i) for i in range(1, min(row + 1, 8 - col))],
                          [(row - i, col - i) for i in range(1, min(row + 1, col + 1))]]

            # For each direction, take each move until a non empty square is encountered, then stop, since
            # bishops cannot jump over other pieces
            for dst_list in directions:
                for dst in dst_list:
                    dst_color = self.get_color(dst)
                    if dst_color is EMPTY:
                        non_capture_moves.append(Move((row, col), dst, color | piece))
                    elif dst_color is opponent:
                        capture_moves.append(Move((row, col), dst, color | piece))
                        break
                    else:
                        break

        if piece is ROOK or piece is QUEEN:
            # List if each possible move in each of the four directions (right, down, left, up)
            directions = [[(row, i) for i in range(col + 1, 8)],
                          [(i, col) for i in range(row + 1, 8)],
                          [(row, i) for i in range(col - 1, -1, -1)],
                          [(i, col) for i in range(row - 1, -1, -1)]]

            # For each direction, take each move until a non empty square is encountered, then stop, since
            # rooks cannot jump over other pieces
            for dst_list in directions:
                for dst in dst_list:
                    dst_color = self.get_color(dst)
                    if dst_color is EMPTY:
                        non_capture_moves.append(Move((row, col), dst, color | piece))
                    elif dst_color is opponent:
                        capture_moves.append(Move((row, col), dst, color | piece))
                        break
                    else:
                        break

        if piece is KING:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (i != 0 or j != 0) and 0 <= row+i < 8 and 0 <= col+j < 8:
                        dst_color = self.get_color((row+i, col+j))
                        if dst_color is EMPTY:
                            non_capture_moves.append(Move((row, col), (row+i, col+j), color | piece))
                        elif dst_color is opponent:
                            capture_moves.append(Move((row, col), (row+i, col+j), color | piece))

            # Castling options
            if not self.king_moved[color] and not self.rook_moved[color][0]:
                if self.board[row][1] is EMPTY and self.board[row][2] is EMPTY and self.board[row][3] is EMPTY:
                    # Cannot move from, or through check in castling move
                    if not remove_check:        # If we don't care about moving into check, add move
                        non_capture_moves.append(Move((row, col), (row, 2), color | piece))
                    # Otherwise check if move will create check, if it doesn't add move
                    elif not self.is_check() and \
                            not self.move(Move((row, col), (row, 3), color | piece)).is_check(color):
                        non_capture_moves.append(Move((row, col), (row, 2), color | piece))

            if not self.king_moved[color] and not self.rook_moved[color][1]:
                if self.board[row][5] is EMPTY and self.board[row][6] is EMPTY:
                    # Cannot move from, or through check in castling move
                    if not remove_check:        # If we don't care about moving into check, add move
                        non_capture_moves.append(Move((row, col), (row, 2), color | piece))
                    # Otherwise check if move will create check, if it doesn't add move
                    elif not self.is_check() and \
                            not self.move(Move((row, col), (row, 5), color | piece)).is_check(color):
                        non_capture_moves.append(Move((row, col), (row, 6), color | piece))

        if remove_check:
            # Remove any moves that put the player in check
            for i in range(len(capture_moves)-1, -1, -1):
                if self.move(capture_moves[i]).is_check(color):
                    del capture_moves[i]

            for i in range(len(non_capture_moves)-1, -1, -1):
                if self.move(non_capture_moves[i]).is_check(color):
                    del non_capture_moves[i]

        return capture_moves, non_capture_moves

    def get_all_moves(self, color=None, remove_check=True):
        """
        Finds and returns a list of all the moves the given color can currently make

        Args:
            color (int): the color to find moves for (defaults to the player whose turn it is)
            remove_check (bool): whether moves that result in check should be removed. Ordinarily, these moves are
                illegal and so should be removed. However, when determining if the other player is in check, these
                moves must be considered. Thus, if getting moves to check if a player is in check, set remove_check to
                False so that these moves are not removed.
        Returns:
            list<Move>: A list of every possible move the given color can make
        """
        if color is None:
            color = self.to_move
        if self.capture_moves[color] is not None:
            return self.capture_moves[color], self.non_capture_moves[color]
        capture_moves = []
        non_capture_moves = []
        for row in range(8):
            for col in range(8):
                if self.get_color((row, col)) is color:
                    capture, non_capture = self.get_moves((row, col), remove_check=remove_check)
                    capture_moves += capture
                    non_capture_moves += non_capture

        self.capture_moves[color] = capture_moves
        self.non_capture_moves[color] = non_capture_moves
        return capture_moves, non_capture_moves


class Move:
    """
    Represents a single move. src and dst are tuples containing the coordinates of the source and destination of the
    move. These tuples are in the form (row, col). final_piece is the piece that will be in the dst location after
    the move is completed. Typically, this is equal to the piece that was initially in the src location, however,
    for pawn promotion this is not the case.
    """

    def __init__(self, src, dst, final_piece):
        self.src = src
        self.dst = dst
        self.final_piece = final_piece

    def __str__(self):
        return '({}, {}{} to {}{})'.format(NAMES[self.final_piece & 0x7], chr(self.src[1] + ord('a')), 8 - self.src[0],
                                           chr(self.dst[1] + ord('a')), 8 - self.dst[0])

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if type(other) is not Move:
            return False

        return self.src == other.src and self.dst == other.dst and self.final_piece == other.final_piece


class ChessUI:
    """
    Attributes:
        scr:
        state (ChessState):
        selection (tuple):
    """
    def __init__(self, scr, state):
        self.scr = scr
        self.selection = (4, 4)
        self.state = state

    def minimax(self, state, depth, alpha, beta, random_move=True):
        """
        Args:
            alpha (int):
            beta (int):
            random_move (bool):
            state (ChessState):
            depth (int):
        Returns:
            (int, Move):
        """
        if depth == 0 or state.is_game_finished():
            return state.value(), None

        capture_moves, non_capture_moves = state.get_all_moves()
        moves = capture_moves + non_capture_moves
        potential_moves = []
        # white wants to maximize the value -> find maximum
        if state.to_move is WHITE:
            maximum = -1001
            for move in moves:
                value, _ = self.minimax(state.move(move), depth-1, alpha, beta, random_move=False)
                alpha = max(alpha, value)
                if random_move and value == maximum:
                    potential_moves.append(move)
                elif value > maximum:
                    maximum = value
                    potential_moves = [move]
                if alpha >= beta:
                    break
            move = random.choice(potential_moves) if random_move else moves[0]
            return maximum, move

        # black wants to minimize the value -> find minimum
        if state.to_move is BLACK:
            minimum = 1001
            for move in moves:
                value, _ = self.minimax(state.move(move), depth - 1, alpha, beta, random_move=False)
                beta = min(beta, value)
                if random_move and value == minimum:
                    potential_moves.append(move)
                elif value < minimum:
                    minimum = value
                    potential_moves = [move]
                if alpha >= beta:
                    break
            move = random.choice(potential_moves) if random_move else moves[0]
            return minimum, move

    def computer_move(self):
        """Chooses a move for the computer to play, and performs that move"""
        value, move = self.minimax(self.state, 3, -1001, 1001)
        self.state = self.state.move(move)
        if self.state.is_checkmate():
            self.display_checkmate_message()

        if self.state.is_stalemate():
            self.display_stalemate_message()

        return move

    def display_quit_prompt(self):
        """Displays a quit prompt and quits if the user selects yes. If not, simply returns"""
        yes = False
        # clear side screen
        for i in range(1, 9):
            self.scr.addstr(i, 27, ' '*12)

        while True:
            self.scr.addstr(2, 27, 'Are you sure')
            self.scr.addstr(3, 29, 'you wish')
            self.scr.addstr(4, 29, 'to quit?')
            if yes:
                self.scr.addstr(6, 28, '    Yes   ', curses.color_pair(YELLOW))
                self.scr.addstr(7, 28, '    No    ')
            else:
                self.scr.addstr(6, 28, '    Yes   ')
                self.scr.addstr(7, 28, '    No    ', curses.color_pair(YELLOW))
            key = self.scr.getch()
            if key == curses.KEY_DOWN:
                yes = False
            if key == curses.KEY_UP:
                yes = True
            if key in [10, 13]:
                if yes:
                    raise SystemExit

                # clear side screen
                for i in range(1, 9):
                    self.scr.addstr(i, 27, ' ' * 12)

                return

    def display_checkmate_message(self):
        """
        Informs the user that checkmate as occurred and exits after the user presses a key
        """
        self.state.draw_board(self.scr)
        # Draw the offending move
        capture_moves, _ = self.state.get_all_moves(WHITE if self.state.to_move is BLACK else BLACK, remove_check=False)
        for move in capture_moves:
            # An opponent move is able to capture the king if this is true
            if self.state.get_color(move.dst) is self.state.to_move and self.state.get_piece(move.dst) is KING:
                self.state.draw_move(self.scr, move)

        # clear side screen
        for i in range(1, 9):
            self.scr.addstr(i, 27, ' ' * 12)

        self.scr.addstr(2, 28, f'{NAMES[self.state.to_move]} is in'.capitalize())
        self.scr.addstr(3, 29, 'checkmate')
        self.scr.addstr(4, 28, f'{NAMES[WHITE if self.state.to_move is BLACK else BLACK]} Wins!'.capitalize())
        self.scr.addstr(6, 29, 'Press any')
        self.scr.addstr(7, 28, 'key to quit')
        self.scr.getch()
        raise SystemExit

    def display_stalemate_message(self):
        """Informs the user that stalemate as occurred and exits after the user presses a key"""
        # clear side screen
        for i in range(1, 9):
            self.scr.addstr(i, 27, ' ' * 12)

        self.state.draw_board(self.scr)
        self.scr.addstr(3, 29, 'Stalemate')
        self.scr.addstr(6, 29, 'Press any')
        self.scr.addstr(7, 28, 'key to quit')
        self.scr.getch()
        raise SystemExit

    def human_move(self):
        """
        Allows the user to select the move they wish to make, performs that move, then performs the oppositions move.
        The oppositions move is returned, so that it can be displayed

        Returns:

        """
        while True:
            self.state.draw_board(self.scr)
            self.state.draw_possible_moves(self.scr, self.selection, None)
            self.state.draw_captured_pieces(self.scr)

            key = self.scr.getch()
            if key == 127:
                return None
            if key == ord('q'):
                self.display_quit_prompt()
            if key == curses.KEY_LEFT:
                self.selection = get_adjacent_square(self.selection, (0, -1))
            if key == curses.KEY_RIGHT:
                self.selection = get_adjacent_square(self.selection, (0, 1))
            if key == curses.KEY_UP:
                self.selection = get_adjacent_square(self.selection, (-1, 0))
            if key == curses.KEY_DOWN:
                self.selection = get_adjacent_square(self.selection, (1, 0))
            if key in [10, 13]:
                capture_moves, non_capture_moves = self.state.get_moves(self.selection)
                if len(capture_moves) + len(non_capture_moves) > 0 and \
                        self.state.get_color(self.selection) is self.state.to_move:
                    move = self.get_move_destination()
                    if move is not None:
                        return move
                else:
                    curses.beep()

    def get_move_destination(self):
        """
        Allows the user to select the destination of the move they want to make, and returns the result as a move
        object. If no destination is selected, then None is returned
        Returns:

        """
        capture_moves, non_capture_moves = self.state.get_moves(self.selection)
        moves = capture_moves + non_capture_moves
        move = moves[0]
        while True:
            self.state.draw_board(self.scr)
            self.state.draw_possible_moves(self.scr, self.selection, move.dst)
            self.state.draw_captured_pieces(self.scr)
            key = self.scr.getch()
            if key == 127:
                return
            if key == ord('q'):
                self.display_quit_prompt()
            if key == curses.KEY_LEFT:
                move = get_adjacent_move(moves, move, (0, -1))
            if key == curses.KEY_RIGHT:
                move = get_adjacent_move(moves, move, (0, 1))
            if key == curses.KEY_UP:
                move = get_adjacent_move(moves, move, (-1, 0))
            if key == curses.KEY_DOWN:
                move = get_adjacent_move(moves, move, (1, 0))
            if key in [10, 13]:
                if self.state.is_pawn_promotion_move(move):
                    result = self.get_pawn_promotion_piece(move)
                    # clear side screen
                    for i in range(1, 9):
                        self.scr.addstr(i, 27, ' ' * 12)
                    if result is None:
                        continue
                    move = result
                self.state = self.state.move(move)
                if self.state.is_checkmate():
                    self.display_checkmate_message()

                if self.state.is_stalemate():
                    self.display_stalemate_message()

                return move

    def get_pawn_promotion_piece(self, move):
        color = self.state.get_color(move.src)
        possible_moves = [Move(move.src, move.dst, color | QUEEN), Move(move.src, move.dst, color | ROOK),
                          Move(move.src, move.dst, color | BISHOP), Move(move.src, move.dst, color | KNIGHT)]
        chosen_move = 0

        # clear side screen
        for i in range(1, 9):
            self.scr.addstr(i, 27, ' ' * 12)

        self.scr.addstr(2, 31, 'Pawn')
        self.scr.addstr(3, 28, 'Promotion')
        self.scr.addstr(5, 27, 'Select piece')
        self.scr.addstr(6, 28, 'to promote')
        self.scr.addstr(7, 32, 'to')
        while True:
            chosen_state = self.state.move(possible_moves[chosen_move])
            chosen_state.draw_board(self.scr)
            chosen_state.draw_square(self.scr, move.dst, MAGENTA)

            key = self.scr.getch()
            if key == 127:
                return None
            if key == ord('q'):
                self.display_quit_prompt()
            if key == curses.KEY_UP:
                chosen_move = (chosen_move + 1) % 4
            if key == curses.KEY_DOWN:
                chosen_move = (chosen_move - 1) % 4
            if key in [10, 13]:
                return possible_moves[chosen_move]


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        self.print_usage(sys.stderr)
        message = message.replace('humans', 'humans/color')
        self.exit(2, f'{self.prog}: error: {message}\n')


def get_adjacent_move(moves, curr, direction):
    """
    Given a list of moves, finds the move that should be moved to for a given direction.

    Args:
        moves: The list of possible moves. All of these should start from the same location
        curr (Move): The currently selected move
        direction (tuple): The direction in which to move the selected square
    Returns:
        tuple: The move that should be selected next given the direction
    """
    # Finds the closest square within the given direction specified. For example if row=1 col=0, finds the
    # closest square that is below the current secondary selection, and sets that to be the secondary selection
    closest_move = None
    closest_dist = 1000
    for move in moves:
        # This condition checks if the moves destination is to the left/right/above/below to the current
        # selection based on what sign rows and col are
        if (move.dst[0] - curr.dst[0]) * direction[0] > 0 or (move.dst[1] - curr.dst[1]) * direction[1] > 0:
            dist = (curr.dst[0] - move.dst[0]) ** 2 + (curr.dst[1] - move.dst[1]) ** 2
            if dist < closest_dist:
                closest_dist = dist
                closest_move = move

    if closest_move is not None:
        return closest_move
    return curr


def get_adjacent_square(pos, direction):
    """
    Returns the square adjacent to pos, in the direction provided by the direction argument. Both arguments are integer
    tuples of the form (row, col). If the adjacent square would be off the chess board, then the original pos is
    returned
    Args:
        pos (tuple): The location of the square to find the adjacent square
        direction (tuple): The direction the adjacent square should be in
    Returns:
        (tuple) The adjacent square to the pos in the direction provided by the direction argument
    """
    adj = (pos[0] + direction[0], pos[1] + direction[1])
    if 0 <= adj[0] < 8 and 0 <= adj[1] < 8:
        return adj
    return pos


# GRAY
# WHITE
# BLACK
# RED
# GREEN
# BLUE
# YELLOW


def main(stdscr):

    state = ChessState()
    ui = ChessUI(stdscr, state)
    move = None

    stdscr.clear()
    # ADD 8 TO GET BRIGHT VERSIONS OF COLORS
    curses.init_pair(BOARD_COLOR, TEXT_COLOR, curses.COLOR_WHITE + 8)
    curses.init_pair(BOARD_COLOR + 1, TEXT_COLOR, curses.COLOR_BLACK + 8)
    curses.init_pair(GREEN, TEXT_COLOR, curses.COLOR_GREEN + 8)
    curses.init_pair(GREEN + 1, TEXT_COLOR, curses.COLOR_GREEN)
    curses.init_pair(YELLOW, TEXT_COLOR, curses.COLOR_YELLOW + 8)
    curses.init_pair(YELLOW + 1, TEXT_COLOR, curses.COLOR_YELLOW)
    curses.init_pair(RED, TEXT_COLOR, curses.COLOR_RED + 8)
    curses.init_pair(RED + 1, TEXT_COLOR, curses.COLOR_RED)
    curses.init_pair(CYAN, TEXT_COLOR, curses.COLOR_CYAN + 8)
    curses.init_pair(CYAN + 1, TEXT_COLOR, curses.COLOR_CYAN)
    curses.init_pair(BLUE, TEXT_COLOR, curses.COLOR_BLUE + 8)
    curses.init_pair(BLUE + 1, TEXT_COLOR, curses.COLOR_BLUE)
    curses.init_pair(MAGENTA, TEXT_COLOR, curses.COLOR_MAGENTA + 8)
    curses.init_pair(MAGENTA + 1, TEXT_COLOR, curses.COLOR_MAGENTA)
    curses.curs_set(False)

    ui.state.draw_board(stdscr)
    ui.state.draw_captured_pieces(stdscr)
    stdscr.refresh()

    while True:
        ui.state.draw_board(stdscr)
        ui.state.draw_move(stdscr, move)
        ui.state.draw_captured_pieces(stdscr)
        if player_types[ui.state.to_move] == 'human':
            key = stdscr.getch()
            if key == ord('q'):
                ui.display_quit_prompt()
                continue
            next_move = ui.human_move()
            if next_move is not None:
                move = next_move
        else:
            stdscr.refresh()
            move = ui.computer_move()


if __name__ == '__main__':
    parser = ArgumentParser(description='Chess game')
    parser.add_argument('args', choices=['0', '1', '2', 'black', 'white'], nargs='?', default='1',
                        help='The number of human players. Valid values are 0, 1, and 2. '
                             'Defaults to 1 (single player) if not provided', metavar='humans')
    parser.add_argument('color', choices=['black', 'white'], nargs='?', default=None,
                        help="The color of the human player, when playing single player. Valid values are 'white' and "
                             "'black'. If this argument is included, the humans argument must either be 1, or be "
                             "omitted. Defaults to 'white' if not included", metavar='color')

    args = parser.parse_args()

    black = 'computer'
    white = 'computer'
    if args.args == '0':
        if args.color is not None:
            parser.error(f'unrecognized arguments: {args.color}')
    if args.args == '1':
        if args.color == 'black':
            black = 'human'
        else:
            white = 'human'
    if args.args == '2':
        if args.color is not None:
            parser.error(f'unrecognized arguments: {args.color}')
        black = 'human'
        white = 'human'
    if args.args == 'black':
        if args.color is not None:
            parser.error(f'unrecognized arguments: {args.color}')
        black = 'human'
        white = 'computer'
    if args.args == 'white':
        if args.color is not None:
            parser.error(f'unrecognized arguments: {args.color}')
        black = 'computer'
        white = 'human'

    player_types = {BLACK: black, WHITE: white}

    wrapper(main)
