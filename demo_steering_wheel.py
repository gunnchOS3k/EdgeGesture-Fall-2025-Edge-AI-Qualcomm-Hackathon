#!/usr/bin/env python3
"""
Demo script showing the virtual steering wheel concept
This demonstrates how hand angles map to steering wheel rotation
"""

import pygame
import math
import time

class SteeringWheelDemo:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Virtual Steering Wheel Demo")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Steering wheel properties
        self.steering_wheel_center = (self.width // 2, self.height // 2)
        self.steering_wheel_radius = 100
        self.steering_angle = 0
        self.max_steering_angle = 45
        
        # Demo angle (simulating hand rotation)
        self.demo_angle = 0
        self.angle_speed = 1
        
        # Colors
        self.colors = {
            'background': (20, 20, 20),
            'steering_wheel': (50, 50, 50),
            'steering_wheel_rim': (200, 200, 200),
            'text': (255, 255, 255),
            'angle_text': (255, 255, 0)
        }
    
    def update(self):
        """Update demo angle."""
        self.demo_angle += self.angle_speed
        
        # Map demo angle to steering angle
        # Simulate hand angle range (-180 to 180) to steering range (-45 to 45)
        self.steering_angle = max(-self.max_steering_angle, 
                                min(self.max_steering_angle, self.demo_angle * 0.25))
        
        # Reverse direction when reaching limits
        if abs(self.demo_angle) > 180:
            self.angle_speed = -self.angle_speed
    
    def draw(self):
        """Draw the steering wheel demo."""
        self.screen.fill(self.colors['background'])
        
        center_x, center_y = self.steering_wheel_center
        
        # Draw steering wheel background
        pygame.draw.circle(self.screen, self.colors['steering_wheel'], 
                         (center_x, center_y), self.steering_wheel_radius + 10)
        
        # Draw steering wheel rim
        pygame.draw.circle(self.screen, self.colors['steering_wheel_rim'], 
                         (center_x, center_y), self.steering_wheel_radius, 5)
        
        # Draw steering wheel spokes
        for angle in [0, 120, 240]:
            spoke_angle = math.radians(angle + self.steering_angle)
            end_x = center_x + math.cos(spoke_angle) * (self.steering_wheel_radius - 20)
            end_y = center_y + math.sin(spoke_angle) * (self.steering_wheel_radius - 20)
            pygame.draw.line(self.screen, self.colors['steering_wheel_rim'], 
                           (center_x, center_y), (end_x, end_y), 4)
        
        # Center hub
        pygame.draw.circle(self.screen, self.colors['steering_wheel'], 
                         (center_x, center_y), 15)
        
        # Draw angle indicators
        self.draw_angle_indicators()
        
        # Draw text information
        self.draw_text()
    
    def draw_angle_indicators(self):
        """Draw angle indicators around the steering wheel."""
        center_x, center_y = self.steering_wheel_center
        
        # Draw angle markers
        for angle in range(-45, 46, 15):
            angle_rad = math.radians(angle)
            start_x = center_x + math.cos(angle_rad) * (self.steering_wheel_radius + 20)
            start_y = center_y + math.sin(angle_rad) * (self.steering_wheel_radius + 20)
            end_x = center_x + math.cos(angle_rad) * (self.steering_wheel_radius + 30)
            end_y = center_y + math.sin(angle_rad) * (self.steering_wheel_radius + 30)
            
            pygame.draw.line(self.screen, self.colors['steering_wheel_rim'], 
                           (start_x, start_y), (end_x, end_y), 2)
        
        # Draw current angle indicator
        current_angle_rad = math.radians(self.steering_angle)
        indicator_x = center_x + math.cos(current_angle_rad) * (self.steering_wheel_radius + 40)
        indicator_y = center_y + math.sin(current_angle_rad) * (self.steering_wheel_radius + 40)
        pygame.draw.circle(self.screen, (255, 0, 0), (int(indicator_x), int(indicator_y)), 5)
    
    def draw_text(self):
        """Draw text information."""
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)
        
        # Title
        title = font_large.render("Virtual Steering Wheel Demo", True, self.colors['text'])
        title_rect = title.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title, title_rect)
        
        # Current angles
        demo_text = font_medium.render(f"Demo Angle: {self.demo_angle:.1f}°", True, self.colors['text'])
        self.screen.blit(demo_text, (50, 100))
        
        steering_text = font_medium.render(f"Steering Angle: {self.steering_angle:.1f}°", True, self.colors['angle_text'])
        self.screen.blit(steering_text, (50, 140))
        
        # Control mapping
        controls = [
            "Control Mapping:",
            f"Hand Angle: {self.demo_angle:.1f}° → Steering: {self.steering_angle:.1f}°",
            "",
            "In the actual game:",
            "• Rotate hand left/right → Steer car",
            "• Open hand → Accelerate (W)",
            "• Make fist → Brake (S)",
            "",
            "Press ESC to quit"
        ]
        
        y_offset = 200
        for line in controls:
            text = font_small.render(line, True, self.colors['text'])
            self.screen.blit(text, (50, y_offset))
            y_offset += 25
    
    def run(self):
        """Main demo loop."""
        print("Virtual Steering Wheel Demo")
        print("=========================")
        print("This demonstrates how hand angles map to steering wheel rotation")
        print("Press ESC to quit")
        print()
        
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                
                self.update()
                self.draw()
                pygame.display.flip()
                self.clock.tick(60)
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            pygame.quit()
            print("Demo completed")

if __name__ == "__main__":
    demo = SteeringWheelDemo()
    demo.run()
