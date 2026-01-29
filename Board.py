"""Board class for Connect 4 game logic."""
import numpy as np


class Board:
    """Represents a Connect 4 board with game logic."""
    
    ROWS = 6
    COLS = 7
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2
    
    def __init__(self):
        """Initialize an empty board."""
        self.grid = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
    
    def reset(self):
        """Clear the board for a new game."""
        self.grid = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
    
    def copy(self):
        """Return a deep copy of the board."""
        new_board = Board()
        new_board.grid = self.grid.copy()
        return new_board
    
    def get_state(self):
        """Return the board state as a numpy array."""
        return self.grid.copy()
    
    def is_valid_move(self, column):
        """Check if a piece can be dropped in the specified column."""
        if column < 0 or column >= self.COLS:
            return False
        # Column is valid if the top row is empty
        return self.grid[0, column] == self.EMPTY
    
    def get_valid_moves(self):
        """Return a list of columns that can accept a piece."""
        return [col for col in range(self.COLS) if self.is_valid_move(col)]
    
    def drop_piece(self, column, player):
        """
        Drop a piece in the specified column.
        
        Args:
            column: The column to drop the piece in (0-6)
            player: The player making the move (1 or 2)
            
        Returns:
            The row where the piece landed, or -1 if invalid move
        """
        if not self.is_valid_move(column):
            return -1
        
        # Find the lowest empty row in the column
        for row in range(self.ROWS - 1, -1, -1):
            if self.grid[row, column] == self.EMPTY:
                self.grid[row, column] = player
                return row
        
        return -1
    
    def check_winner(self):
        """
        Check if there's a winner.
        
        Returns:
            1 if Player 1 wins, 2 if Player 2 wins, None otherwise
        """
        # Check horizontal
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                if self.grid[row, col] != self.EMPTY:
                    if (self.grid[row, col] == self.grid[row, col + 1] == 
                        self.grid[row, col + 2] == self.grid[row, col + 3]):
                        return self.grid[row, col]
        
        # Check vertical
        for row in range(self.ROWS - 3):
            for col in range(self.COLS):
                if self.grid[row, col] != self.EMPTY:
                    if (self.grid[row, col] == self.grid[row + 1, col] == 
                        self.grid[row + 2, col] == self.grid[row + 3, col]):
                        return self.grid[row, col]
        
        # Check diagonal (bottom-left to top-right)
        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                if self.grid[row, col] != self.EMPTY:
                    if (self.grid[row, col] == self.grid[row - 1, col + 1] == 
                        self.grid[row - 2, col + 2] == self.grid[row - 3, col + 3]):
                        return self.grid[row, col]
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                if self.grid[row, col] != self.EMPTY:
                    if (self.grid[row, col] == self.grid[row + 1, col + 1] == 
                        self.grid[row + 2, col + 2] == self.grid[row + 3, col + 3]):
                        return self.grid[row, col]
        
        return None
    
    def is_full(self):
        """Check if the board is completely full (draw condition)."""
        return len(self.get_valid_moves()) == 0
    
    def is_terminal(self):
        """Check if the game is over (win or draw)."""
        return self.check_winner() is not None or self.is_full()
    
    def __str__(self):
        """String representation of the board for debugging."""
        symbols = {self.EMPTY: '.', self.PLAYER1: 'X', self.PLAYER2: 'O'}
        lines = []
        for row in range(self.ROWS):
            line = ' '.join(symbols[cell] for cell in self.grid[row])
            lines.append(line)
        lines.append('0 1 2 3 4 5 6')  # Column numbers
        return '\n'.join(lines)
