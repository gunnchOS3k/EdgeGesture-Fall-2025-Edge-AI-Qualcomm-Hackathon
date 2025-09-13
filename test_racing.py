#!/usr/bin/env python3
"""
Test script for the racing game without camera
"""

import pygame
import sys
from racing_game import RacingGame

def test_racing_game():
    """Test the racing game with keyboard controls."""
    print("Testing Racing Game...")
    print("Controls:")
    print("- W: Accelerate")
    print("- S: Brake")
    print("- A: Turn Left")
    print("- D: Turn Right")
    print("- ESC: Quit")
    print()
    
    game = RacingGame()
    
    if not game.initialize():
        print("Failed to initialize racing game")
        return
    
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            dt = clock.tick(60) / 1000.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_w:
                        game.gesture_controls['w_pressed'] = True
                    elif event.key == pygame.K_s:
                        game.gesture_controls['s_pressed'] = True
                    elif event.key == pygame.K_a:
                        game.gesture_controls['a_pressed'] = True
                    elif event.key == pygame.K_d:
                        game.gesture_controls['d_pressed'] = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        game.gesture_controls['w_pressed'] = False
                    elif event.key == pygame.K_s:
                        game.gesture_controls['s_pressed'] = False
                    elif event.key == pygame.K_a:
                        game.gesture_controls['a_pressed'] = False
                    elif event.key == pygame.K_d:
                        game.gesture_controls['d_pressed'] = False
            
            # Update and draw
            game.update(dt)
            game.draw()
            pygame.display.flip()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        game.cleanup()
        print("Test completed")

if __name__ == "__main__":
    test_racing_game()
