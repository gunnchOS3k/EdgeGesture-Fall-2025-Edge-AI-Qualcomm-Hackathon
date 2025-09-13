"""
Base game class for the multi-game gesture control system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pygame
import sys
import os

class BaseGame(ABC):
    """Base class for all games in the gesture control system."""
    
    def __init__(self, width: int = 1200, height: int = 800):
        """Initialize the base game."""
        self.width = width
        self.height = height
        self.screen = None
        self.clock = None
        self.running = False
        self.game_name = "Base Game"
        
        # Gesture control state
        self.gesture_controls = {}
        self.last_gesture_data = {}
        
    def initialize(self) -> bool:
        """Initialize the game. Returns True if successful."""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.game_name)
            self.clock = pygame.time.Clock()
            self.running = True
            return True
        except Exception as e:
            print(f"Error initializing {self.game_name}: {e}")
            return False
    
    def cleanup(self):
        """Clean up game resources."""
        if self.screen:
            pygame.quit()
    
    def handle_gesture(self, gesture_data: Dict[str, Any]) -> None:
        """
        Handle gesture input from the hand detector.
        
        Args:
            gesture_data: Dictionary containing gesture information
                - 'index_finger_pos': (x, y) position of index finger
                - 'is_fist': Boolean for fist gesture
                - 'hand_angle': Angle of hand rotation (for steering wheel)
                - 'screen_region': 'left', 'middle', 'right'
        """
        self.last_gesture_data = gesture_data
        self._process_gesture_input(gesture_data)
    
    @abstractmethod
    def _process_gesture_input(self, gesture_data: Dict[str, Any]) -> None:
        """Process gesture input specific to each game."""
        pass
    
    @abstractmethod
    def update(self, dt: float) -> None:
        """Update game logic."""
        pass
    
    @abstractmethod
    def draw(self) -> None:
        """Draw the game."""
        pass
    
    def run(self) -> None:
        """Main game loop."""
        if not self.initialize():
            return
        
        try:
            while self.running:
                dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    self.handle_event(event)
                
                # Update and draw
                self.update(dt)
                self.draw()
                pygame.display.flip()
                
        except KeyboardInterrupt:
            print(f"{self.game_name} interrupted by user")
        finally:
            self.cleanup()
    
    def handle_event(self, event) -> None:
        """Handle pygame events. Override in subclasses if needed."""
        pass
    
    def get_gesture_controls(self) -> Dict[str, Any]:
        """Get current gesture control state."""
        return self.gesture_controls
