#!/usr/bin/env python3
"""
Racing Game Launcher with Gesture Control
Runs the racing game and camera simultaneously
"""

import cv2
import numpy as np
import pygame
import threading
import time
from hand_detector import HandDetector
from racing_game import RacingGame

class RacingLauncher:
    def __init__(self):
        """Initialize the racing game launcher."""
        self.hand_detector = HandDetector()
        self.racing_game = RacingGame()
        self.cap = None
        self.running = False
        
    def initialize_camera(self) -> bool:
        """Initialize the webcam."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def camera_thread(self):
        """Camera processing thread."""
        print("Camera thread started")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand and get gesture information
            index_finger_pos, is_fist, annotated_frame, hand_angle = self.hand_detector.detect_hand(frame)
            
            # Determine screen region
            screen_region = 'middle'  # Default
            if index_finger_pos:
                x_pos = index_finger_pos[0]
                screen_region = self.hand_detector.get_screen_region(x_pos, frame.shape[1])
            
            # Send gesture data to racing game
            gesture_data = {
                'index_finger_pos': index_finger_pos,
                'is_fist': is_fist,
                'hand_angle': hand_angle,
                'screen_region': screen_region
            }
            self.racing_game.handle_gesture(gesture_data)
            
            # Draw debug overlay
            debug_frame = self.draw_debug_overlay(
                annotated_frame, index_finger_pos, screen_region, is_fist, hand_angle
            )
            
            # Display the frame
            cv2.imshow('Gesture Controller - Racing', debug_frame)
            
            # Check for quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
        
        print("Camera thread ended")
    
    def draw_debug_overlay(self, frame: np.ndarray, index_finger_pos: tuple, 
                          screen_region: str, is_fist: bool, hand_angle: float = 0.0) -> np.ndarray:
        """Draw debug overlay on the frame."""
        h, w = frame.shape[:2]
        
        # Draw screen thirds (vertical lines for left/right)
        third = w // 3
        cv2.line(frame, (third, 0), (third, h), (0, 255, 0), 2)
        cv2.line(frame, (2 * third, 0), (2 * third, h), (0, 255, 0), 2)
        
        # Add region labels
        cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "MIDDLE", (third + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "RIGHT", (2 * third + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw index finger position
        if index_finger_pos:
            x, y = index_finger_pos
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Finger: ({x}, {y})", (x + 15, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw current region and gesture info
        region_color = (0, 255, 0) if screen_region != 'middle' else (255, 255, 255)
        cv2.putText(frame, f"Region: {screen_region.upper()}", (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_color, 2)
        
        # Draw hand angle for steering wheel
        cv2.putText(frame, f"Hand Angle: {hand_angle:.1f}°", (w - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if is_fist:
            cv2.putText(frame, "FIST DETECTED", (w - 200, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Rotate hand left/right to steer", (10, h - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Open hand to accelerate (W)", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Make fist to brake (S)", (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main application loop."""
        print("Racing Game with Gesture Control")
        print("===============================")
        print("Make sure you have a clear view of your hand!")
        print("Press 'q' in the camera window to quit")
        print()
        
        if not self.initialize_camera():
            return
        
        # Initialize pygame for the racing game
        if not self.racing_game.initialize():
            print("Failed to initialize racing game")
            return
        
        self.running = True
        
        try:
            # Start camera thread
            camera_thread = threading.Thread(target=self.camera_thread)
            camera_thread.daemon = True
            camera_thread.start()
            
            # Run racing game in main thread
            self.racing_game.run()
            
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.running = False
        
        if self.racing_game:
            self.racing_game.cleanup()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    """Main entry point."""
    launcher = RacingLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
