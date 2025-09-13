#!/usr/bin/env python3
"""
Dual Hand Racing Controller
Shows steering wheel in camera view and uses both hands for control
"""

import cv2
import numpy as np
import math
import pyautogui
import time
from hand_detector import HandDetector

class DualHandRacingController:
    def __init__(self):
        """Initialize the dual hand racing controller."""
        self.hand_detector = HandDetector()
        self.cap = None
        self.running = False
        
        # Steering wheel properties
        self.steering_wheel_center = (320, 240)  # Center of 640x480 camera
        self.steering_wheel_radius = 80
        self.steering_angle = 0
        self.max_steering_angle = 45
        
        # Hand tracking
        self.left_hand_angle = 0
        self.right_hand_angle = 0
        self.left_hand_fist = False
        self.right_hand_fist = False
        
        # Control state
        self.current_keys = set()
        self.last_key_time = 0
        self.key_delay = 0.05
        
        # Colors
        self.colors = {
            'steering_wheel': (50, 50, 50),
            'steering_wheel_rim': (200, 200, 200),
            'steering_wheel_center': (100, 100, 100),
            'hand_left': (0, 255, 0),
            'hand_right': (255, 0, 0),
            'text': (255, 255, 255),
            'control_active': (0, 255, 0),
            'control_inactive': (100, 100, 100)
        }
    
    def initialize_camera(self) -> bool:
        """Initialize the webcam."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_dual_hands(self, frame):
        """Detect both hands and return their information."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hand_detector.hands.process(rgb_frame)
        
        left_hand_data = None
        right_hand_data = None
        
        if results.multi_hand_landmarks:
            # Get frame dimensions
            h, w, _ = frame.shape
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Determine if this is left or right hand
                # Use wrist and middle finger to determine hand orientation
                wrist = hand_landmarks.landmark[0]
                middle_base = hand_landmarks.landmark[9]
                
                # Calculate hand center
                hand_center_x = (wrist.x + middle_base.x) / 2
                
                # Determine hand side based on position
                if hand_center_x < 0.5:  # Left side of screen
                    left_hand_data = self.extract_hand_data(hand_landmarks, w, h)
                else:  # Right side of screen
                    right_hand_data = self.extract_hand_data(hand_landmarks, w, h)
        
        return left_hand_data, right_hand_data
    
    def extract_hand_data(self, hand_landmarks, width, height):
        """Extract hand data from landmarks."""
        landmarks = hand_landmarks.landmark
        
        # Get hand center
        wrist = landmarks[0]
        middle_base = landmarks[9]
        hand_center = (
            int((wrist.x + middle_base.x) * width / 2),
            int((wrist.y + middle_base.y) * height / 2)
        )
        
        # Calculate hand angle
        wrist_x = wrist.x * width
        wrist_y = wrist.y * height
        middle_x = middle_base.x * width
        middle_y = middle_base.y * height
        
        dx = middle_x - wrist_x
        dy = middle_y - wrist_y
        angle_rad = math.atan2(dy, dx)
        hand_angle = math.degrees(angle_rad)
        
        # Normalize angle
        if hand_angle > 180:
            hand_angle -= 360
        elif hand_angle < -180:
            hand_angle += 360
        
        # Detect fist
        is_fist = self.hand_detector._detect_fist(hand_landmarks, width, height)
        
        return {
            'center': hand_center,
            'angle': hand_angle,
            'is_fist': is_fist
        }
    
    def calculate_steering_angle(self, left_hand, right_hand):
        """Calculate steering angle from both hands."""
        if not left_hand or not right_hand:
            return 0
        
        # Average the angles of both hands
        avg_angle = (left_hand['angle'] + right_hand['angle']) / 2
        
        # Map to steering angle (-45 to 45 degrees)
        steering_angle = max(-self.max_steering_angle, 
                           min(self.max_steering_angle, avg_angle * 0.5))
        
        return steering_angle
    
    def update_controls(self, left_hand, right_hand):
        """Update keyboard controls based on hand gestures."""
        current_time = time.time()
        
        if current_time - self.last_key_time < self.key_delay:
            return
        
        # Calculate steering angle
        self.steering_angle = self.calculate_steering_angle(left_hand, right_hand)
        
        # Determine which keys should be pressed
        target_keys = set()
        
        # Steering controls (A/D)
        if self.steering_angle < -5:  # Turn left
            target_keys.add('a')
        elif self.steering_angle > 5:  # Turn right
            target_keys.add('d')
        
        # Acceleration controls (W/S)
        if left_hand and right_hand:
            left_fist = left_hand['is_fist']
            right_fist = right_hand['is_fist']
            
            if not left_fist and not right_fist:
                # Both hands open = accelerate
                target_keys.add('w')
            elif left_fist and right_fist:
                # Both hands closed = brake
                target_keys.add('s')
            # Mixed states = coast (no W or S)
        
        # Update key presses
        self.update_key_presses(target_keys)
        self.last_key_time = current_time
    
    def update_key_presses(self, target_keys):
        """Update which keys are pressed."""
        # Release keys that are no longer needed
        keys_to_release = self.current_keys - target_keys
        for key in keys_to_release:
            try:
                pyautogui.keyUp(key)
                print(f"Released key: {key}")
            except Exception as e:
                print(f"Error releasing key {key}: {e}")
        
        # Press keys that are needed but not currently pressed
        keys_to_press = target_keys - self.current_keys
        for key in keys_to_press:
            try:
                pyautogui.keyDown(key)
                print(f"Pressed key: {key}")
            except Exception as e:
                print(f"Error pressing key {key}: {e}")
        
        # Update current keys
        self.current_keys = target_keys
    
    def draw_steering_wheel(self, frame):
        """Draw the steering wheel on the frame."""
        center_x, center_y = self.steering_wheel_center
        
        # Steering wheel background
        cv2.circle(frame, (center_x, center_y), self.steering_wheel_radius + 10, 
                  self.colors['steering_wheel'], -1)
        
        # Steering wheel rim
        cv2.circle(frame, (center_x, center_y), self.steering_wheel_radius, 
                  self.colors['steering_wheel_rim'], 3)
        
        # Steering wheel spokes
        for angle in [0, 120, 240]:
            spoke_angle = math.radians(angle + self.steering_angle)
            end_x = int(center_x + math.cos(spoke_angle) * (self.steering_wheel_radius - 20))
            end_y = int(center_y + math.sin(spoke_angle) * (self.steering_wheel_radius - 20))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), 
                    self.colors['steering_wheel_rim'], 3)
        
        # Center hub
        cv2.circle(frame, (center_x, center_y), 15, self.colors['steering_wheel_center'], -1)
        
        # Steering angle indicator
        angle_text = f"{self.steering_angle:.1f}°"
        cv2.putText(frame, angle_text, (center_x - 30, center_y + self.steering_wheel_radius + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
    
    def draw_hand_info(self, frame, left_hand, right_hand):
        """Draw hand information on the frame."""
        h, w = frame.shape[:2]
        
        # Draw hand positions and angles
        if left_hand:
            center = left_hand['center']
            angle = left_hand['angle']
            is_fist = left_hand['is_fist']
            
            # Draw hand center
            cv2.circle(frame, center, 10, self.colors['hand_left'], -1)
            
            # Draw hand info
            cv2.putText(frame, f"L: {angle:.1f}° {'FIST' if is_fist else 'OPEN'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['hand_left'], 2)
        
        if right_hand:
            center = right_hand['center']
            angle = right_hand['angle']
            is_fist = right_hand['is_fist']
            
            # Draw hand center
            cv2.circle(frame, center, 10, self.colors['hand_right'], -1)
            
            # Draw hand info
            cv2.putText(frame, f"R: {angle:.1f}° {'FIST' if is_fist else 'OPEN'}", 
                       (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['hand_right'], 2)
    
    def draw_controls(self, frame):
        """Draw control status on the frame."""
        h, w = frame.shape[:2]
        
        # Control status
        y_offset = 60
        for key in ['w', 'a', 's', 'd']:
            pressed = key in self.current_keys
            color = self.colors['control_active'] if pressed else self.colors['control_inactive']
            status = "ON" if pressed else "OFF"
            
            cv2.putText(frame, f"{key.upper()}: {status}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Instructions
        instructions = [
            "Dual Hand Racing Controls:",
            "• Hold steering wheel with both hands",
            "• Rotate hands to steer (A/D)",
            "• Both hands open = Accelerate (W)",
            "• Both hands closed = Brake (S)",
            "• Mixed = Coast",
            "",
            "Press 'q' to quit"
        ]
        
        y_offset = h - 200
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += 20
    
    def run(self):
        """Main application loop."""
        print("Dual Hand Racing Controller")
        print("==========================")
        print("Hold the steering wheel with both hands!")
        print("Press 'q' to quit")
        print()
        
        if not self.initialize_camera():
            return
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect both hands
                left_hand, right_hand = self.detect_dual_hands(frame)
                
                # Update controls
                self.update_controls(left_hand, right_hand)
                
                # Draw steering wheel
                self.draw_steering_wheel(frame)
                
                # Draw hand information
                self.draw_hand_info(frame, left_hand, right_hand)
                
                # Draw controls
                self.draw_controls(frame)
                
                # Display the frame
                cv2.imshow('Dual Hand Racing Controller', frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        # Release all keys
        for key in self.current_keys.copy():
            try:
                pyautogui.keyUp(key)
                print(f"Released key: {key}")
            except Exception as e:
                print(f"Error releasing key {key}: {e}")
        
        self.current_keys.clear()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    """Main entry point."""
    controller = DualHandRacingController()
    controller.run()

if __name__ == "__main__":
    main()
