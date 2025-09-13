#!/usr/bin/env python3
"""
Proper Racing Controller with Correct Gesture Logic and Calibration
"""

import cv2
import numpy as np
import math
import pyautogui
import time
from hand_detector import HandDetector

class ProperRacingController:
    def __init__(self):
        """Initialize the proper racing controller."""
        self.hand_detector = HandDetector()
        self.cap = None
        self.running = False
        
        # Camera dimensions
        self.camera_width = 640
        self.camera_height = 480
        
        # Calibration
        self.calibrated_center_angle = None
        self.calibration_complete = False
        
        # Steering wheel properties
        self.steering_wheel_center = (self.camera_width // 2, self.camera_height // 2)
        self.steering_wheel_radius = 100
        
        # Control thresholds
        self.steering_deadzone = 15  # degrees
        self.steering_threshold = 20  # degrees
        
        # Hand tracking
        self.left_hand_pos = None
        self.right_hand_pos = None
        self.left_hand_gripping = False
        self.right_hand_gripping = False
        self.both_hands_open = False
        
        # Control state
        self.current_keys = set()
        self.last_key_time = 0
        self.key_delay = 0.03
        
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
            'line': (255, 0, 255)
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
    
    def detect_hands_proper(self, frame):
        """Detect both hands with proper positioning."""
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
                hand_data = self.extract_hand_data_proper(hand_landmarks, w, h)
                
                # Determine hand side based on position relative to center
                if hand_data['center'][0] < w // 2:
                    left_hand_data = hand_data
                else:
                    right_hand_data = hand_data
        
        return left_hand_data, right_hand_data
    
    def extract_hand_data_proper(self, hand_landmarks, width, height):
        """Extract hand data with proper grip detection."""
        landmarks = hand_landmarks.landmark
        
        # Get hand center (palm center)
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        index_mcp = landmarks[5]
        
        # Calculate hand center
        hand_center = (
            int((wrist.x + middle_mcp.x + index_mcp.x) * width / 3),
            int((wrist.y + middle_mcp.y + index_mcp.y) * height / 3)
        )
        
        # Detect if hand is gripping (closed around wheel)
        is_gripping = self.detect_grip(hand_landmarks, width, height)
        
        # Detect if hand is open (flat palm for braking)
        is_open = self.detect_open_palm(hand_landmarks, width, height)
        
        return {
            'center': hand_center,
            'is_gripping': is_gripping,
            'is_open': is_open
        }
    
    def detect_grip(self, hand_landmarks, width, height):
        """Detect if hand is gripping the wheel (closed around it)."""
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
            if distance < 30:
                closed_fingers += 1
        
        # Consider it gripping if 3 or more fingers are closed
        return closed_fingers >= 3
    
    def detect_open_palm(self, hand_landmarks, width, height):
        """Detect if hand is open (flat palm for braking)."""
        landmarks = hand_landmarks.landmark
        
        # Check if fingertips are far from their respective knuckles
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        open_fingers = 0
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_pos = (landmarks[tip_idx].x * width, landmarks[tip_idx].y * height)
            pip_pos = (landmarks[pip_idx].x * width, landmarks[pip_idx].y * height)
            
            # Calculate distance between tip and PIP
            distance = math.sqrt((tip_pos[0] - pip_pos[0])**2 + (tip_pos[1] - pip_pos[1])**2)
            
            # If distance is large, finger is likely open
            if distance > 40:
                open_fingers += 1
        
        # Consider it open if 4 or more fingers are open
        return open_fingers >= 4
    
    def calculate_wheel_angle(self, left_hand, right_hand):
        """Calculate the angle of the line between both hands."""
        if not left_hand or not right_hand:
            return None
        
        left_pos = left_hand['center']
        right_pos = right_hand['center']
        
        # Calculate angle of line between hands
        dx = right_pos[0] - left_pos[0]
        dy = right_pos[1] - left_pos[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to -180 to 180
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360
        
        return angle_deg
    
    def calibrate_center(self, left_hand, right_hand):
        """Calibrate the center angle for steering."""
        if not left_hand or not right_hand:
            return False
        
        current_angle = self.calculate_wheel_angle(left_hand, right_hand)
        if current_angle is not None:
            self.calibrated_center_angle = current_angle
            self.calibration_complete = True
            print(f"Calibrated center angle: {self.calibrated_center_angle:.1f}°")
            return True
        return False
    
    def update_controls_proper(self, left_hand, right_hand):
        """Update controls with proper logic."""
        current_time = time.time()
        
        if current_time - self.last_key_time < self.key_delay:
            return
        
        # Update hand positions and states
        self.left_hand_pos = left_hand['center'] if left_hand else None
        self.right_hand_pos = right_hand['center'] if right_hand else None
        self.left_hand_gripping = left_hand['is_gripping'] if left_hand else False
        self.right_hand_gripping = right_hand['is_gripping'] if right_hand else False
        self.both_hands_open = (left_hand and left_hand['is_open'] and 
                              right_hand and right_hand['is_open'])
        
        # Determine which keys should be pressed
        target_keys = set()
        
        # Acceleration (W) - both hands gripping
        if self.left_hand_gripping and self.right_hand_gripping:
            target_keys.add('w')
        
        # Braking (S) - both hands open
        if self.both_hands_open:
            target_keys.add('s')
        
        # Steering (A/D) - based on wheel angle offset
        if self.calibration_complete and left_hand and right_hand:
            current_angle = self.calculate_wheel_angle(left_hand, right_hand)
            if current_angle is not None:
                angle_offset = current_angle - self.calibrated_center_angle
                
                # Normalize angle offset to -180 to 180
                if angle_offset > 180:
                    angle_offset -= 360
                elif angle_offset < -180:
                    angle_offset += 360
                
                # Apply steering logic
                if angle_offset < -self.steering_threshold:
                    target_keys.add('a')  # Turn left
                elif angle_offset > self.steering_threshold:
                    target_keys.add('d')  # Turn right
                # Within deadzone = no steering
        
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
    
    def draw_steering_wheel(self, frame):
        """Draw the steering wheel."""
        center_x, center_y = self.steering_wheel_center
        
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
        
        # Center hub
        cv2.circle(frame, (center_x, center_y), 20, self.colors['steering_wheel_center'], -1)
    
    def draw_hands_and_line(self, frame, left_hand, right_hand):
        """Draw hands and the line between them."""
        # Draw hands
        if left_hand:
            center = left_hand['center']
            color = self.colors['hand_left']
            cv2.circle(frame, center, 15, color, -1)
            cv2.circle(frame, center, 20, color, 3)
            
            # Draw grip status
            status = "GRIP" if left_hand['is_gripping'] else "OPEN" if left_hand['is_open'] else "NEUTRAL"
            cv2.putText(frame, f"L: {status}", (center[0] - 30, center[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if right_hand:
            center = right_hand['center']
            color = self.colors['hand_right']
            cv2.circle(frame, center, 15, color, -1)
            cv2.circle(frame, center, 20, color, 3)
            
            # Draw grip status
            status = "GRIP" if right_hand['is_gripping'] else "OPEN" if right_hand['is_open'] else "NEUTRAL"
            cv2.putText(frame, f"R: {status}", (center[0] - 30, center[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw line between hands
        if left_hand and right_hand:
            cv2.line(frame, left_hand['center'], right_hand['center'], 
                    self.colors['line'], 3)
            
            # Draw current angle
            current_angle = self.calculate_wheel_angle(left_hand, right_hand)
            if current_angle is not None:
                angle_text = f"Angle: {current_angle:.1f}°"
                cv2.putText(frame, angle_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
                
                # Draw angle offset if calibrated
                if self.calibration_complete:
                    angle_offset = current_angle - self.calibrated_center_angle
                    if angle_offset > 180:
                        angle_offset -= 360
                    elif angle_offset < -180:
                        angle_offset += 360
                    
                    offset_text = f"Offset: {angle_offset:.1f}°"
                    cv2.putText(frame, offset_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
    
    def draw_controls(self, frame):
        """Draw control status."""
        h, w = frame.shape[:2]
        
        # Control status
        y_offset = h - 150
        for key in ['w', 'a', 's', 'd']:
            pressed = key in self.current_keys
            color = self.colors['control_active'] if pressed else self.colors['control_inactive']
            status = "ON" if pressed else "OFF"
            
            cv2.putText(frame, f"{key.upper()}: {status}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30
        
        # Instructions
        if not self.calibration_complete:
            instructions = [
                "CALIBRATION NEEDED",
                "Place both hands on the wheel",
                "Press 'c' to calibrate center position",
                "Then grip to accelerate, open to brake"
            ]
        else:
            instructions = [
                "RACING CONTROLS:",
                "• Grip both hands = Accelerate (W)",
                "• Open both hands = Brake (S)",
                "• Rotate wheel = Steer (A/D)",
                "• Press 'c' to recalibrate",
                "• Press 'q' to quit"
            ]
        
        y_offset = 10
        for instruction in instructions:
            cv2.putText(frame, instruction, (w - 300, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            y_offset += 25
    
    def run(self):
        """Main application loop."""
        print("Proper Racing Controller")
        print("=======================")
        print("Place both hands on the wheel and press 'c' to calibrate")
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
                left_hand, right_hand = self.detect_hands_proper(frame)
                
                # Update controls if calibrated
                if self.calibration_complete and left_hand and right_hand:
                    self.update_controls_proper(left_hand, right_hand)
                
                # Draw steering wheel
                self.draw_steering_wheel(frame)
                
                # Draw hands and line
                self.draw_hands_and_line(frame, left_hand, right_hand)
                
                # Draw controls
                self.draw_controls(frame)
                
                # Display the frame
                cv2.imshow('Proper Racing Controller', frame)
                
                # Check for commands
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('c'):
                    # Calibrate
                    if self.calibrate_center(left_hand, right_hand):
                        print("Calibration complete!")
                    else:
                        print("Calibration failed - make sure both hands are visible")
        
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
    controller = ProperRacingController()
    controller.run()

if __name__ == "__main__":
    main()
