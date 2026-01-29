"""Graphics engine for Connect 4 visualization using Pygame."""
import pygame
import numpy as np


class Engine:
    """Pygame-based graphics engine for rendering the Connect 4 board."""
    
    # Colors (RGB)
    BACKGROUND = (40, 44, 52)
    BOARD_COLOR = (0, 102, 204)
    EMPTY_SLOT = (30, 34, 42)
    PLAYER1_COLOR = (231, 76, 60)    # Red
    PLAYER2_COLOR = (241, 196, 15)   # Yellow
    HIGHLIGHT = (255, 255, 255)
    
    # Dimensions
    CELL_SIZE = 100
    PIECE_RADIUS = 42
    PADDING = 10
    
    def __init__(self, board, window_title="Connect 4"):
        """
        Initialize the graphics engine.
        
        Args:
            board: The Board instance to render
            window_title: Title for the game window
        """
        self.board = board
        
        # Calculate window dimensions
        self.width = board.COLS * self.CELL_SIZE
        self.height = (board.ROWS + 1) * self.CELL_SIZE  # Extra row for piece preview
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(window_title)
        self.clock = pygame.time.Clock()
        
        # Font initialization (may fail on some Python versions)
        try:
            self.font = pygame.font.Font(None, 36)
        except (NotImplementedError, ImportError):
            self.font = None
        
        # Hover column for preview (-1 = none)
        self.hover_col = -1
        self.current_player = 1
    
    def set_hover_column(self, col):
        """Set which column is being hovered over for preview."""
        self.hover_col = col
    
    def set_current_player(self, player):
        """Set the current player for hover preview color."""
        self.current_player = player
    
    def get_column_from_mouse(self, x):
        """Convert mouse x position to column index."""
        col = x // self.CELL_SIZE
        if 0 <= col < self.board.COLS:
            return col
        return -1
    
    def render(self):
        """Draw the current board state to the screen."""
        # Clear background
        self.screen.fill(self.BACKGROUND)
        
        # Draw hover preview piece (in the top row)
        if self.hover_col >= 0 and self.board.is_valid_move(self.hover_col):
            color = self.PLAYER1_COLOR if self.current_player == 1 else self.PLAYER2_COLOR
            center_x = self.hover_col * self.CELL_SIZE + self.CELL_SIZE // 2
            center_y = self.CELL_SIZE // 2
            pygame.draw.circle(self.screen, color, (center_x, center_y), self.PIECE_RADIUS)
        
        # Draw the board (blue rectangle with holes)
        board_rect = pygame.Rect(0, self.CELL_SIZE, self.width, self.board.ROWS * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.BOARD_COLOR, board_rect)
        
        # Draw cells and pieces
        for row in range(self.board.ROWS):
            for col in range(self.board.COLS):
                center_x = col * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = (row + 1) * self.CELL_SIZE + self.CELL_SIZE // 2
                
                # Determine color based on cell content
                cell_value = self.board.grid[row, col]
                if cell_value == self.board.EMPTY:
                    color = self.EMPTY_SLOT
                elif cell_value == self.board.PLAYER1:
                    color = self.PLAYER1_COLOR
                else:
                    color = self.PLAYER2_COLOR
                
                # Draw the piece/slot
                pygame.draw.circle(self.screen, color, (center_x, center_y), self.PIECE_RADIUS)
        
        # Update display
        pygame.display.flip()
    
    def render_game_over(self, winner):
        """Render game over overlay."""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Determine winner color for visual indicator
        if winner is None:
            color = self.HIGHLIGHT
        else:
            color = self.PLAYER1_COLOR if winner == 1 else self.PLAYER2_COLOR
        
        # Draw a large circle in the center as visual indicator
        pygame.draw.circle(self.screen, color, (self.width // 2, self.height // 2), 60)
        pygame.draw.circle(self.screen, self.HIGHLIGHT, (self.width // 2, self.height // 2), 60, 4)
        
        # Render text if font is available
        if self.font:
            if winner is None:
                text = "Draw!"
            else:
                text = f"Player {winner} Wins!"
            
            text_surface = self.font.render(text, True, color)
            text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2 - 20))
            self.screen.blit(text_surface, text_rect)
            
            restart_text = self.font.render("Press R to restart, Q to quit", True, self.HIGHLIGHT)
            restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def get_frame(self):
        """
        Capture the current frame as a numpy array.
        
        Returns:
            RGB numpy array of shape (height, width, 3)
        """
        # Get the raw pixel data from the screen
        frame = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), we want (height, width, 3)
        frame = np.transpose(frame, (1, 0, 2))
        return frame
    
    def get_frame_resized(self, target_size=(84, 84)):
        """
        Get frame resized for neural network input.
        
        Args:
            target_size: Tuple of (height, width) for output
            
        Returns:
            RGB numpy array of shape (target_size[0], target_size[1], 3)
        """
        frame = self.get_frame()
        # Use pygame to resize
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled = pygame.transform.scale(surface, (target_size[1], target_size[0]))
        result = pygame.surfarray.array3d(scaled)
        return np.transpose(result, (1, 0, 2))
    
    def handle_events(self):
        """
        Process pygame events.
        
        Returns:
            dict with keys:
                - 'quit': True if window closed or Q pressed
                - 'reset': True if R pressed
                - 'click_col': Column clicked (-1 if no click)
                - 'hover_col': Column being hovered over
        """
        result = {
            'quit': False,
            'reset': False,
            'click_col': -1,
            'hover_col': -1
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result['quit'] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    result['quit'] = True
                elif event.key == pygame.K_r:
                    result['reset'] = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    result['click_col'] = self.get_column_from_mouse(event.pos[0])
        
        # Always track hover position
        mouse_x, _ = pygame.mouse.get_pos()
        result['hover_col'] = self.get_column_from_mouse(mouse_x)
        self.hover_col = result['hover_col']
        
        return result
    
    def tick(self, fps=60):
        """Limit frame rate and return delta time."""
        return self.clock.tick(fps)
    
    def quit(self):
        """Clean up pygame resources."""
        pygame.quit()
