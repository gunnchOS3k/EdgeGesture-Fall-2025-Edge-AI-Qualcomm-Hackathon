"""
Racing Game with Virtual Steering Wheel
Controls: Gesture-based steering wheel for WASD controls
"""

import pygame
import math
import random
from typing import Dict, Any, Tuple, List
from game_base import BaseGame

class RacingGame(BaseGame):
    """3D-style racing game with virtual steering wheel controls."""
    
    def __init__(self, width: int = 1200, height: int = 800):
        super().__init__(width, height)
        self.game_name = "Racing Game - Gesture Controlled"
        
        # Game state
        self.car_x = width // 2
        self.car_y = height - 100
        self.car_angle = 0
        self.car_speed = 0
        self.max_speed = 8
        self.acceleration = 0.3
        self.deceleration = 0.2
        self.turn_speed = 0.1
        
        # Road parameters
        self.road_width = 400
        self.road_x = width // 2 - self.road_width // 2
        self.road_curves = []
        self.road_offset = 0
        
        # Steering wheel
        self.steering_wheel_center = (width - 150, height - 150)
        self.steering_wheel_radius = 60
        self.steering_angle = 0
        self.max_steering_angle = 45  # degrees
        
        # Gesture controls
        self.gesture_controls = {
            'w_pressed': False,
            'a_pressed': False,
            's_pressed': False,
            'd_pressed': False
        }
        
        # Visual elements
        self.colors = {
            'road': (60, 60, 60),
            'road_marking': (255, 255, 255),
            'grass': (34, 139, 34),
            'car': (255, 0, 0),
            'steering_wheel': (50, 50, 50),
            'steering_wheel_rim': (200, 200, 200),
            'speedometer': (255, 255, 255)
        }
        
        # Initialize road curves
        self._generate_road_curves()
    
    def _generate_road_curves(self):
        """Generate random road curves for more interesting gameplay."""
        for i in range(100):
            curve = random.uniform(-0.02, 0.02)
            self.road_curves.append(curve)
    
    def _process_gesture_input(self, gesture_data: Dict[str, Any]) -> None:
        """Process gesture input for steering wheel controls."""
        # Get hand angle for steering
        hand_angle = gesture_data.get('hand_angle', 0)
        is_fist = gesture_data.get('is_fist', False)
        index_finger_pos = gesture_data.get('index_finger_pos')
        
        # Calculate steering angle based on hand rotation
        # Map hand angle (-180 to 180) to steering angle (-45 to 45 degrees)
        self.steering_angle = max(-self.max_steering_angle, 
                                min(self.max_steering_angle, hand_angle * 0.25))
        
        # Map steering angle to WASD controls
        self._update_wasd_controls()
        
        # Fist gesture for brake (S key)
        if is_fist:
            self.gesture_controls['s_pressed'] = True
        else:
            self.gesture_controls['s_pressed'] = False
        
        # Open hand for acceleration (W key) - when not making a fist
        if not is_fist and index_finger_pos:
            self.gesture_controls['w_pressed'] = True
        else:
            self.gesture_controls['w_pressed'] = False
    
    def _update_wasd_controls(self):
        """Update WASD controls based on steering angle."""
        # Reset all controls
        for key in ['a_pressed', 'd_pressed']:
            self.gesture_controls[key] = False
        
        # Map steering angle to A/D controls
        if self.steering_angle < -5:  # Turn left
            self.gesture_controls['a_pressed'] = True
        elif self.steering_angle > 5:  # Turn right
            self.gesture_controls['d_pressed'] = True
    
    def update(self, dt: float) -> None:
        """Update game logic."""
        # Handle acceleration/deceleration
        if self.gesture_controls['w_pressed'] and not self.gesture_controls['s_pressed']:
            self.car_speed = min(self.max_speed, self.car_speed + self.acceleration)
        elif self.gesture_controls['s_pressed']:
            self.car_speed = max(-self.max_speed * 0.5, self.car_speed - self.deceleration)
        else:
            # Natural deceleration
            if self.car_speed > 0:
                self.car_speed = max(0, self.car_speed - self.deceleration * 0.5)
            elif self.car_speed < 0:
                self.car_speed = min(0, self.car_speed + self.deceleration * 0.5)
        
        # Handle steering
        if self.car_speed != 0:
            if self.gesture_controls['a_pressed']:  # Turn left
                self.car_angle -= self.turn_speed * abs(self.car_speed) * 0.1
            elif self.gesture_controls['d_pressed']:  # Turn right
                self.car_angle += self.turn_speed * abs(self.car_speed) * 0.1
        
        # Update car position
        if self.car_speed != 0:
            self.car_x += math.cos(self.car_angle) * self.car_speed
            self.car_y += math.sin(self.car_angle) * self.car_speed
        
        # Keep car on screen
        self.car_x = max(50, min(self.width - 50, self.car_x))
        self.car_y = max(50, min(self.height - 50, self.car_y))
        
        # Update road offset for scrolling effect
        self.road_offset += self.car_speed * 0.1
    
    def draw(self) -> None:
        """Draw the game."""
        self.screen.fill(self.colors['grass'])
        
        # Draw road
        self._draw_road()
        
        # Draw car
        self._draw_car()
        
        # Draw steering wheel
        self._draw_steering_wheel()
        
        # Draw HUD
        self._draw_hud()
    
    def _draw_road(self):
        """Draw the racing road with 3D perspective."""
        # Simple road with perspective
        road_top_width = self.road_width
        road_bottom_width = self.road_width + 200
        
        # Road top
        road_top_left = self.road_x
        road_top_right = self.road_x + road_top_width
        
        # Road bottom (wider for perspective)
        road_bottom_left = self.road_x - 100
        road_bottom_right = self.road_x + road_bottom_width
        
        # Draw road surface
        road_points = [
            (road_top_left, 0),
            (road_top_right, 0),
            (road_bottom_right, self.height),
            (road_bottom_left, self.height)
        ]
        pygame.draw.polygon(self.screen, self.colors['road'], road_points)
        
        # Draw road markings
        self._draw_road_markings()
    
    def _draw_road_markings(self):
        """Draw road center line and side markings."""
        # Center line
        center_x = self.width // 2
        for y in range(0, self.height, 40):
            pygame.draw.rect(self.screen, self.colors['road_marking'], 
                           (center_x - 2, y, 4, 20))
        
        # Side markings
        road_left = self.road_x
        road_right = self.road_x + self.road_width
        
        for y in range(0, self.height, 30):
            # Left side
            pygame.draw.rect(self.screen, self.colors['road_marking'], 
                           (road_left - 5, y, 3, 15))
            # Right side
            pygame.draw.rect(self.screen, self.colors['road_marking'], 
                           (road_right + 2, y, 3, 15))
    
    def _draw_car(self):
        """Draw the racing car."""
        # Car body
        car_rect = pygame.Rect(self.car_x - 20, self.car_y - 10, 40, 20)
        pygame.draw.rect(self.screen, self.colors['car'], car_rect)
        
        # Car direction indicator
        front_x = self.car_x + math.cos(self.car_angle) * 25
        front_y = self.car_y + math.sin(self.car_angle) * 25
        pygame.draw.circle(self.screen, (255, 255, 0), (int(front_x), int(front_y)), 5)
    
    def _draw_steering_wheel(self):
        """Draw the virtual steering wheel."""
        center_x, center_y = self.steering_wheel_center
        
        # Steering wheel background
        pygame.draw.circle(self.screen, self.colors['steering_wheel'], 
                         (center_x, center_y), self.steering_wheel_radius + 5)
        
        # Steering wheel rim
        pygame.draw.circle(self.screen, self.colors['steering_wheel_rim'], 
                         (center_x, center_y), self.steering_wheel_radius, 3)
        
        # Steering wheel spokes
        for angle in [0, 120, 240]:
            spoke_angle = math.radians(angle + self.steering_angle)
            end_x = center_x + math.cos(spoke_angle) * (self.steering_wheel_radius - 10)
            end_y = center_y + math.sin(spoke_angle) * (self.steering_wheel_radius - 10)
            pygame.draw.line(self.screen, self.colors['steering_wheel_rim'], 
                           (center_x, center_y), (end_x, end_y), 3)
        
        # Center hub
        pygame.draw.circle(self.screen, self.colors['steering_wheel'], 
                         (center_x, center_y), 8)
        
        # Steering angle indicator
        angle_text = f"{self.steering_angle:.1f}°"
        font = pygame.font.Font(None, 24)
        text_surface = font.render(angle_text, True, self.colors['speedometer'])
        text_rect = text_surface.get_rect(center=(center_x, center_y + self.steering_wheel_radius + 30))
        self.screen.blit(text_surface, text_rect)
    
    def _draw_hud(self):
        """Draw heads-up display."""
        # Speedometer
        speed_text = f"Speed: {abs(self.car_speed):.1f}"
        font = pygame.font.Font(None, 36)
        text_surface = font.render(speed_text, True, self.colors['speedometer'])
        self.screen.blit(text_surface, (20, 20))
        
        # Control indicators
        controls_y = 60
        for key, pressed in self.gesture_controls.items():
            color = (0, 255, 0) if pressed else (100, 100, 100)
            key_text = key.upper().replace('_PRESSED', '')
            font = pygame.font.Font(None, 24)
            text_surface = font.render(f"{key_text}: {'ON' if pressed else 'OFF'}", True, color)
            self.screen.blit(text_surface, (20, controls_y))
            controls_y += 25
        
        # Instructions
        instructions = [
            "Gesture Controls:",
            "• Rotate hand left/right to steer",
            "• Open hand to accelerate (W)",
            "• Make fist to brake (S)",
            "• A/D keys controlled by steering angle"
        ]
        
        font = pygame.font.Font(None, 20)
        for i, instruction in enumerate(instructions):
            text_surface = font.render(instruction, True, self.colors['speedometer'])
            self.screen.blit(text_surface, (self.width - 300, 20 + i * 25))
