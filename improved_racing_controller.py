#!/usr/bin/env python3
"""
Improved Racing Controller with Better Calibration and Clear Controls
"""

import cv2
import numpy as np
import math
import pyautogui
import time
from hand_detector import HandDetector

class ImprovedRacingController:
    def __init__(self):
        """Initialize the improved racing controller."""
        self.hand_detector = HandDetector()
        self.cap = None
        self.running = False
        
        # Camera dimensions
        self.camera_width = 640
        self.camera_height = 480
        
        # Steering wheel properties
        self.steering_wheel_center = (self.camera_width // 2, self.camera_height // 2)
        self.steering_wheel_radius = 100
        self.steering_angle = 0
        self.max_steering_angle = 45
        
        # Calibration
        self.calibration_mode = True
        self.left_hand_center = None
        self.right_hand_center = None
        self.calibration_complete = False
        
        # Hand tracking
        self.left_hand_data = None
        self.right_hand_data = None
        
        # Control state
        self.current_keys = set()
        self.last_key_time = 0
        self.key_delay = 0.03  # Faster response
        
        # Colors
        self.colors = {
            'steering_wheel': (40, 40, 40),
            'steering_wheel_rim': (255, 255, 255),
            'steering_wheel_center': (60, 60, 60),
            'hand_left': (0, 255, 0),
            'hand_right': (0, 0, 255),
            'text': (255, 255, 255),
            'control_active': (0, 255, 0),
            'control_inactive': (100, 100, 100),
            'calibration': (255, 255, 0),
            'guidelines': (100, 100, 255)
        }
    
    def initialize_camera(self) -> bool:
        """Initialize the webcam."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_hands_improved(self, frame):
        """Improved hand detection with better positioning."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hand_detector.hands.process(rgb_frame)
        
        left_hand_data = None
        right_hand_data = None
        
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand data
                hand_data = self.extract_hand_data_improved(hand_landmarks, w, h)
                
                # Determine hand side based on position relative to center
                if hand_data['center'][0] < w // 2:
                    left_hand_data = hand_data
                else:
                    right_hand_data = hand_data
        
        return left_hand_data, right_hand_data
    
    def extract_hand_data_improved(self, hand_landmarks, width, height):
        """Extract improved hand data from landmarks."""
        landmarks = hand_landmarks.landmark
        
        # Get key points for better angle calculation
        wrist = landmarks[0]
        middle_mcp = landmarks[9]  # Middle finger base
        index_mcp = landmarks[5]   # Index finger base
        
        # Calculate hand center (average of key points)
        hand_center = (
            int((wrist.x + middle_mcp.x + index_mcp.x) * width / 3),
            int((wrist.y + middle_mcp.y + index_mcp.y) * height / 3)
        )
        
        # Calculate hand orientation using wrist to middle finger
        wrist_x = wrist.x * width
        wrist_y = wrist.y * height
        middle_x = middle_mcp.x * width
        middle_y = middle_mcp.y * height
        
        dx = middle_x - wrist_x
        dy = middle_y - wrist_y
        angle_rad = math.atan2(dy, dx)
        hand_angle = math.degrees(angle_rad)
        
        # Normalize angle to -180 to 180
        if hand_angle > 180:
            hand_angle -= 360
        elif hand_angle < -180:
            hand_angle += 360
        
        # Detect fist with improved accuracy
        is_fist = self.detect_fist_improved(hand_landmarks, width, height)
        
        return {
            'center': hand_center,
            'angle': hand_angle,
            'is_fist': is_fist,
            'wrist': (int(wrist.x * width), int(wrist.y * height)),
            'middle_base': (int(middle_x), int(middle_y))
        }
    
    def detect_fist_improved(self, hand_landmarks, width, height):
        """Improved fist detection."""
        landmarks = hand_landmarks.landmark
        
        # Check if fingertips are close to their respective knuckles
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        closed_fingers = 0
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_pos = (landmarks[tip_idx].x * width, landmarks[tip_idx].y * height)
            pip_pos = (landmarks[pip_idx].x * width, landmarks[pip_idx].y * height)
            
            # Calculate distance between tip and PIP
            distance = math.sqrt((tip_pos[0] - pip_pos[0])**2 + (tip_pos[1] - pip_pos[1])**2)
            
            # If distance is small, finger is likely closed
            if distance < 25:  # Adjusted threshold
                closed_fingers += 1
        
        # Consider it a fist if 3 or more fingers are closed
        return closed_fingers >= 3
    
    def calibrate_hands(self, left_hand, right_hand):
        """Calibrate hand positions for better control."""
        if left_hand and right_hand:
            self.left_hand_center = left_hand['center']
            self.right_hand_center = right_hand['center']
            self.calibration_complete = True
            self.calibration_mode = False
            print("Calibration complete! You can now control the steering wheel.")
            return True
        return False
    
    def calculate_steering_angle_improved(self, left_hand, right_hand):
        """Calculate steering angle with improved calibration."""
        if not left_hand or not right_hand or not self.calibration_complete:
            return 0
        
        # Calculate relative angles from calibrated positions
        left_angle = left_hand['angle']
        right_angle = right_hand['angle']
        
        # Average the angles
        avg_angle = (left_angle + right_angle) / 2
        
        # Map to steering angle with better sensitivity
        steering_angle = max(-self.max_steering_angle, 
                           min(self.max_steering_angle, avg_angle * 0.3))
        
        return steering_angle
    
    def update_controls_improved(self, left_hand, right_hand):
        """Update controls with improved logic."""
        current_time = time.time()
        
        if current_time - self.last_key_time < self.key_delay:
            return
        
        # Calculate steering angle
        self.steering_angle = self.calculate_steering_angle_improved(left_hand, right_hand)
        
        # Determine which keys should be pressed
        target_keys = set()
        
        # Steering controls (A/D) - more sensitive
        if self.steering_angle < -3:  # Turn left
            target_keys.add('a')
        elif self.steering_angle > 3:  # Turn right
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
            except Exception as e:
                print(f"Error releasing key {key}: {e}")
        
        # Press keys that are needed but not currently pressed
        keys_to_press = target_keys - self.current_keys
        for key in keys_to_press:
            try:
                pyautogui.keyDown(key)
            except Exception as e:
                print(f"Error pressing key {key}: {e}")
        
        # Update current keys
        self.current_keys = target_keys
    
    def draw_steering_wheel_improved(self, frame):
        """Draw improved steering wheel with guidelines."""
        center_x, center_y = self.steering_wheel_center
        
        # Draw guidelines
        self.draw_guidelines(frame)
        
        # Steering wheel background
        cv2.circle(frame, (center_x, center_y), self.steering_wheel_radius + 15, 
                  self.colors['steering_wheel'], -1)
        
        # Steering wheel rim
        cv2.circle(frame, (center_x, center_y), self.steering_wheel_radius, 
                  self.colors['steering_wheel_rim'], 4)
        
        # Draw angle markers
        for angle in range(-45, 46, 15):
            angle_rad = math.radians(angle)
            start_x = int(center_x + math.cos(angle_rad) * (self.steering_wheel_radius - 10))
            start_y = int(center_y + math.sin(angle_rad) * (self.steering_wheel_radius - 10))
            end_x = int(center_x + math.cos(angle_rad) * (self.steering_wheel_radius + 10))
            end_y = int(center_y + math.sin(angle_rad) * (self.steering_wheel_radius + 10))
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                    self.colors['steering_wheel_rim'], 2)
        
        # Steering wheel spokes
        for angle in [0, 120, 240]:
            spoke_angle = math.radians(angle + self.steering_angle)
            end_x = int(center_x + math.cos(spoke_angle) * (self.steering_wheel_radius - 25))
            end_y = int(center_y + math.sin(spoke_angle) * (self.steering_wheel_radius - 25))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), 
                    self.colors['steering_wheel_rim'], 4)
        
        # Center hub
        cv2.circle(frame, (center_x, center_y), 20, self.colors['steering_wheel_center'], -1)
        
        # Current angle indicator
        indicator_angle = math.radians(self.steering_angle)
        indicator_x = int(center_x + math.cos(indicator_angle) * (self.steering_wheel_radius + 30))
        indicator_y = int(center_y + math.sin(indicator_angle) * (self.steering_wheel_radius + 30))
        cv2.circle(frame, (indicator_x, indicator_y), 8, (0, 255, 0), -1)
        
        # Steering angle text
        angle_text = f"Steering: {self.steering_angle:.1f}°"
        cv2.putText(frame, angle_text, (center_x - 60, center_y + self.steering_wheel_radius + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
    
    def draw_guidelines(self, frame):
        """Draw guidelines for hand positioning."""
        center_x, center_y = self.steering_wheel_center
        
        # Draw hand position zones
        cv2.rectangle(frame, (50, 50), (center_x - 50, center_y - 50), 
                     self.colors['guidelines'], 2)
        cv2.rectangle(frame, (center_x + 50, 50), (self.camera_width - 50, center_y - 50), 
                     self.colors['guidelines'], 2)
        
        # Draw center line
        cv2.line(frame, (center_x, 0), (center_x, self.camera_height), 
                self.colors['guidelines'], 2)
        
        # Draw labels
        cv2.putText(frame, "LEFT HAND", (60, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['guidelines'], 2)
        cv2.putText(frame, "RIGHT HAND", (center_x + 60, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['guidelines'], 2)
    
    def draw_hand_info_improved(self, frame, left_hand, right_hand):
        """Draw improved hand information."""
        h, w = frame.shape[:2]
        
        # Draw hand positions and angles
        if left_hand:
            center = left_hand['center']
            angle = left_hand['angle']
            is_fist = left_hand['is_fist']
            
            # Draw hand center
            cv2.circle(frame, center, 12, self.colors['hand_left'], -1)
            cv2.circle(frame, center, 15, self.colors['hand_left'], 2)
            
            # Draw hand info
            cv2.putText(frame, f"LEFT: {angle:.1f}° {'FIST' if is_fist else 'OPEN'}", 
                       (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['hand_left'], 2)
        
        if right_hand:
            center = right_hand['center']
            angle = right_hand['angle']
            is_fist = right_hand['is_fist']
            
            # Draw hand center
            cv2.circle(frame, center, 12, self.colors['hand_right'], -1)
            cv2.circle(frame, center, 15, self.colors['hand_right'], 2)
            
            # Draw hand info
            cv2.putText(frame, f"RIGHT: {angle:.1f}° {'FIST' if is_fist else 'OPEN'}", 
                       (w - 250, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['hand_right'], 2)
    
    def draw_controls_improved(self, frame):
        """Draw improved control status."""
        h, w = frame.shape[:2]
        
        # Control status
        y_offset = h - 100
        for key in ['w', 'a', 's', 'd']:
            pressed = key in self.current_keys
            color = self.colors['control_active'] if pressed else self.colors['control_inactive']
            status = "ON" if pressed else "OFF"
            
            cv2.putText(frame, f"{key.upper()}: {status}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 25
        
        # Instructions
        if self.calibration_mode:
            instructions = [
                "CALIBRATION MODE",
                "Place both hands in the marked zones",
                "Make sure both hands are visible",
                "Press SPACE when ready"
            ]
        else:
            instructions = [
                "RACING CONTROLS:",
                "• Rotate hands to steer (A/D)",
                "• Both hands OPEN = Accelerate (W)",
                "• Both hands CLOSED = Brake (S)",
                "• Mixed = Coast",
                "",
                "Press 'q' to quit, 'r' to recalibrate"
            ]
        
        y_offset = 10
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            y_offset += 25
    
    def run(self):
        """Main application loop."""
        print("Improved Racing Controller")
        print("=========================")
        print("Place both hands in the marked zones for calibration")
        print("Press 'q' to quit, 'r' to recalibrate")
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
                left_hand, right_hand = self.detect_hands_improved(frame)
                
                # Handle calibration
                if self.calibration_mode:
                    if self.calibrate_hands(left_hand, right_hand):
                        continue
                else:
                    # Update controls
                    self.update_controls_improved(left_hand, right_hand)
                
                # Draw steering wheel
                self.draw_steering_wheel_improved(frame)
                
                # Draw hand information
                self.draw_hand_info_improved(frame, left_hand, right_hand)
                
                # Draw controls
                self.draw_controls_improved(frame)
                
                # Display the frame
                cv2.imshow('Improved Racing Controller', frame)
                
                # Check for commands
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('r'):
                    # Recalibrate
                    self.calibration_mode = True
                    self.calibration_complete = False
                    print("Recalibrating... Place both hands in the marked zones")
                elif key == ord(' ') and self.calibration_mode:
                    # Complete calibration manually
                    if self.calibrate_hands(left_hand, right_hand):
                        continue
        
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
            except Exception as e:
                print(f"Error releasing key {key}: {e}")
        
        self.current_keys.clear()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    """Main entry point."""
    controller = ImprovedRacingController()
    controller.run()

if __name__ == "__main__":
    main()
