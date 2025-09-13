#!/usr/bin/env python3
"""
Racing Controls Demo
Shows exactly how the controls work without camera
"""

import pygame
import math
import time

class RacingControlsDemo:
    def __init__(self):
        pygame.init()
        self.width = 1000
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racing Controls Demo")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Demo state
        self.steering_angle = 0
        self.left_hand_angle = 0
        self.right_hand_angle = 0
        self.left_hand_fist = False
        self.right_hand_fist = False
        
        # Control state
        self.controls = {
            'w': False,  # Accelerate
            'a': False,  # Turn left
            's': False,  # Brake
            'd': False   # Turn right
        }
        
        # Colors
        self.colors = {
            'background': (20, 20, 20),
            'steering_wheel': (50, 50, 50),
            'steering_wheel_rim': (255, 255, 255),
            'hand_left': (0, 255, 0),
            'hand_right': (0, 0, 255),
            'text': (255, 255, 255),
            'control_active': (0, 255, 0),
            'control_inactive': (100, 100, 100)
        }
    
    def update_controls(self):
        """Update controls based on hand simulation."""
        # Calculate steering angle from both hands
        avg_angle = (self.left_hand_angle + self.right_hand_angle) / 2
        self.steering_angle = max(-45, min(45, avg_angle * 0.3))
        
        # Reset controls
        for key in self.controls:
            self.controls[key] = False
        
        # Steering controls
        if self.steering_angle < -3:
            self.controls['a'] = True
        elif self.steering_angle > 3:
            self.controls['d'] = True
        
        # Acceleration controls
        if not self.left_hand_fist and not self.right_hand_fist:
            self.controls['w'] = True  # Both hands open = accelerate
        elif self.left_hand_fist and self.right_hand_fist:
            self.controls['s'] = True  # Both hands closed = brake
        # Mixed states = coast (no W or S)
    
    def draw_steering_wheel(self):
        """Draw the steering wheel."""
        center_x = self.width // 2
        center_y = self.height // 2
        radius = 120
        
        # Steering wheel background
        pygame.draw.circle(self.screen, self.colors['steering_wheel'], 
                         (center_x, center_y), radius + 15)
        
        # Steering wheel rim
        pygame.draw.circle(self.screen, self.colors['steering_wheel_rim'], 
                         (center_x, center_y), radius, 5)
        
        # Draw angle markers
        for angle in range(-45, 46, 15):
            angle_rad = math.radians(angle)
            start_x = center_x + math.cos(angle_rad) * (radius - 15)
            start_y = center_y + math.sin(angle_rad) * (radius - 15)
            end_x = center_x + math.cos(angle_rad) * (radius + 15)
            end_y = center_y + math.sin(angle_rad) * (radius + 15)
            
            pygame.draw.line(self.screen, self.colors['steering_wheel_rim'], 
                           (start_x, start_y), (end_x, end_y), 3)
        
        # Steering wheel spokes
        for angle in [0, 120, 240]:
            spoke_angle = math.radians(angle + self.steering_angle)
            end_x = center_x + math.cos(spoke_angle) * (radius - 30)
            end_y = center_y + math.sin(spoke_angle) * (radius - 30)
            pygame.draw.line(self.screen, self.colors['steering_wheel_rim'], 
                           (center_x, center_y), (end_x, end_y), 5)
        
        # Center hub
        pygame.draw.circle(self.screen, self.colors['steering_wheel'], 
                         (center_x, center_y), 25)
        
        # Current angle indicator
        indicator_angle = math.radians(self.steering_angle)
        indicator_x = center_x + math.cos(indicator_angle) * (radius + 40)
        indicator_y = center_y + math.sin(indicator_angle) * (radius + 40)
        pygame.draw.circle(self.screen, (0, 255, 0), (int(indicator_x), int(indicator_y)), 10)
    
    def draw_hand_simulation(self):
        """Draw hand simulation."""
        # Left hand
        left_x = 200
        left_y = 150
        pygame.draw.circle(self.screen, self.colors['hand_left'], (left_x, left_y), 20)
        pygame.draw.circle(self.screen, self.colors['hand_left'], (left_x, left_y), 25, 3)
        
        # Right hand
        right_x = 800
        right_y = 150
        pygame.draw.circle(self.screen, self.colors['hand_right'], (right_x, right_y), 20)
        pygame.draw.circle(self.screen, self.colors['hand_right'], (right_x, right_y), 25, 3)
        
        # Hand angles
        font = pygame.font.Font(None, 36)
        left_text = font.render(f"Left: {self.left_hand_angle:.1f}°", True, self.colors['hand_left'])
        right_text = font.render(f"Right: {self.right_hand_angle:.1f}°", True, self.colors['hand_right'])
        
        self.screen.blit(left_text, (left_x - 80, left_y + 40))
        self.screen.blit(right_text, (right_x - 80, right_y + 40))
        
        # Fist status
        left_fist_text = font.render("FIST" if self.left_hand_fist else "OPEN", True, self.colors['hand_left'])
        right_fist_text = font.render("FIST" if self.right_hand_fist else "OPEN", True, self.colors['hand_right'])
        
        self.screen.blit(left_fist_text, (left_x - 30, left_y + 70))
        self.screen.blit(right_fist_text, (right_x - 30, right_y + 70))
    
    def draw_controls(self):
        """Draw control status."""
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)
        
        # Title
        title = font_large.render("Racing Controls Demo", True, self.colors['text'])
        self.screen.blit(title, (50, 50))
        
        # Control status
        y_offset = 300
        for key, pressed in self.controls.items():
            color = self.colors['control_active'] if pressed else self.colors['control_inactive']
            status = "ON" if pressed else "OFF"
            
            text = font_medium.render(f"{key.upper()}: {status}", True, color)
            self.screen.blit(text, (50, y_offset))
            y_offset += 40
        
        # Instructions
        instructions = [
            "Controls:",
            "• 1/2: Rotate left hand left/right",
            "• 3/4: Rotate right hand left/right", 
            "• Q/W: Open/close left hand",
            "• E/R: Open/close right hand",
            "• SPACE: Reset angles",
            "• ESC: Quit"
        ]
        
        y_offset = 500
        for instruction in instructions:
            text = font_small.render(instruction, True, self.colors['text'])
            self.screen.blit(text, (50, y_offset))
            y_offset += 25
        
        # Steering angle
        angle_text = font_medium.render(f"Steering Angle: {self.steering_angle:.1f}°", True, self.colors['text'])
        self.screen.blit(angle_text, (self.width - 300, 50))
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_1:  # Left hand left
                    self.left_hand_angle -= 10
                elif event.key == pygame.K_2:  # Left hand right
                    self.left_hand_angle += 10
                elif event.key == pygame.K_3:  # Right hand left
                    self.right_hand_angle -= 10
                elif event.key == pygame.K_4:  # Right hand right
                    self.right_hand_angle += 10
                elif event.key == pygame.K_q:  # Left hand open
                    self.left_hand_fist = False
                elif event.key == pygame.K_w:  # Left hand closed
                    self.left_hand_fist = True
                elif event.key == pygame.K_e:  # Right hand open
                    self.right_hand_fist = False
                elif event.key == pygame.K_r:  # Right hand closed
                    self.right_hand_fist = True
                elif event.key == pygame.K_SPACE:  # Reset
                    self.left_hand_angle = 0
                    self.right_hand_angle = 0
                    self.left_hand_fist = False
                    self.right_hand_fist = False
    
    def run(self):
        """Main demo loop."""
        print("Racing Controls Demo")
        print("===================")
        print("Use keyboard to simulate hand gestures:")
        print("1/2: Rotate left hand left/right")
        print("3/4: Rotate right hand left/right")
        print("Q/W: Open/close left hand")
        print("E/R: Open/close right hand")
        print("SPACE: Reset angles")
        print("ESC: Quit")
        print()
        
        try:
            while self.running:
                self.handle_events()
                self.update_controls()
                
                self.screen.fill(self.colors['background'])
                self.draw_steering_wheel()
                self.draw_hand_simulation()
                self.draw_controls()
                
                pygame.display.flip()
                self.clock.tick(60)
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            pygame.quit()
            print("Demo completed")

if __name__ == "__main__":
    demo = RacingControlsDemo()
    demo.run()
