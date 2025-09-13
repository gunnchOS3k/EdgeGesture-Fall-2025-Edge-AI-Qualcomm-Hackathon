#!/usr/bin/env python3
"""
Robust Simple Racing Controller
Uses proven computer vision techniques for reliable grip detection
"""

import cv2
import numpy as np
import math
import pyautogui
import time
from hand_detector import HandDetector

class RobustSimpleRacingController:
    def __init__(self):
        """Initialize the robust simple racing controller."""
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
        self.steering_deadzone = 10  # degrees
        self.steering_threshold = 15  # degrees
        
        # Hand tracking
        self.left_hand_pos = None
        self.right_hand_pos = None
        self.left_hand_gripping = False
        self.right_hand_gripping = False
        self.both_hands_open = False
        
        # Grip detection with smoothing
        self.left_grip_history = []
        self.right_grip_history = []
        self.grip_history_size = 3
        
        # Control state
        self.current_keys = set()
        self.last_key_time = 0
        self.key_delay = 0.05
        
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
            'line': (255, 0, 255),
            'debug': (255, 255, 0)
        }
    
    def initialize_camera(self) -> bool:
        """Initialize the webcam with proper error handling."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                return False
            
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_hands_robust(self, frame):
        """Robust hand detection."""
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
                hand_data = self.extract_hand_data_robust(hand_landmarks, w, h)
                
                # Determine hand side based on position relative to center
                if hand_data['center'][0] < w // 2:
                    left_hand_data = hand_data
                else:
                    right_hand_data = hand_data
        
        return left_hand_data, right_hand_data
    
    def extract_hand_data_robust(self, hand_landmarks, width, height):
        """Extract hand data with robust grip detection."""
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
        
        # Robust grip detection using multiple methods
        grip_score = self.calculate_grip_score_robust(landmarks, width, height)
        open_score = self.calculate_open_score_robust(landmarks, width, height)
        
        # Determine hand state based on scores - much more sensitive thresholds
        is_gripping = grip_score > 0.3  # Even more sensitive for grip
        is_open = open_score > 0.4
        
        return {
            'center': hand_center,
            'is_gripping': is_gripping,
            'is_open': is_open,
            'grip_score': grip_score,
            'open_score': open_score
        }
    
    def calculate_grip_score_robust(self, landmarks, width, height):
        """Calculate grip score using robust methods."""
        # Method 1: Finger tip to PIP distance
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        closed_fingers = 0
        total_distance = 0
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_pos = (landmarks[tip_idx].x * width, landmarks[tip_idx].y * height)
            pip_pos = (landmarks[pip_idx].x * width, landmarks[pip_idx].y * height)
            
            distance = math.sqrt((tip_pos[0] - pip_pos[0])**2 + (tip_pos[1] - pip_pos[1])**2)
            total_distance += distance
            
            # Much more sensitive threshold for grip detection
            if distance < 35:  # Increased threshold for easier grip detection
                closed_fingers += 1
        
        # Method 2: Thumb to index finger distance
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2
        ) * width
        
        # Method 3: Wrist to middle finger distance
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        wrist_middle_dist = math.sqrt(
            (wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2
        ) * width
        
        # Method 4: Finger tip to palm center distance
        palm_center = landmarks[9]  # Middle finger base
        palm_distances = []
        for tip_idx in finger_tips:
            tip_pos = (landmarks[tip_idx].x * width, landmarks[tip_idx].y * height)
            palm_pos = (palm_center.x * width, palm_center.y * height)
            dist = math.sqrt((tip_pos[0] - palm_pos[0])**2 + (tip_pos[1] - palm_pos[1])**2)
            palm_distances.append(dist)
        
        avg_palm_distance = sum(palm_distances) / len(palm_distances)
        
        # Calculate scores with much more sensitive thresholds
        finger_score = closed_fingers / 5.0
        thumb_score = 1.0 if thumb_index_dist < 60 else 0.0  # More lenient
        wrist_score = 1.0 if wrist_middle_dist < 100 else 0.0  # More lenient
        palm_score = 1.0 if avg_palm_distance < 70 else 0.0  # More lenient
        
        # Weighted combination with palm distance
        grip_score = (finger_score * 0.4 + thumb_score * 0.2 + wrist_score * 0.2 + palm_score * 0.2)
        
        return grip_score
    
    def calculate_open_score_robust(self, landmarks, width, height):
        """Calculate open palm score using robust methods."""
        # Method 1: Finger tip to PIP distance (opposite of grip)
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        open_fingers = 0
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_pos = (landmarks[tip_idx].x * width, landmarks[tip_idx].y * height)
            pip_pos = (landmarks[pip_idx].x * width, landmarks[pip_idx].y * height)
            
            distance = math.sqrt((tip_pos[0] - pip_pos[0])**2 + (tip_pos[1] - pip_pos[1])**2)
            
            # More sensitive threshold for open detection
            if distance > 40:  # Fixed threshold in pixels
                open_fingers += 1
        
        # Method 2: Thumb to index finger distance
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2
        ) * width
        
        # Method 3: Wrist to middle finger distance
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        wrist_middle_dist = math.sqrt(
            (wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2
        ) * width
        
        # Method 4: Finger tip to palm center distance (opposite of grip)
        palm_center = landmarks[9]  # Middle finger base
        palm_distances = []
        for tip_idx in finger_tips:
            tip_pos = (landmarks[tip_idx].x * width, landmarks[tip_idx].y * height)
            palm_pos = (palm_center.x * width, palm_center.y * height)
            dist = math.sqrt((tip_pos[0] - palm_pos[0])**2 + (tip_pos[1] - palm_pos[1])**2)
            palm_distances.append(dist)
        
        avg_palm_distance = sum(palm_distances) / len(palm_distances)
        
        # Calculate scores with more sensitive thresholds
        finger_score = open_fingers / 5.0
        thumb_score = 1.0 if thumb_index_dist > 60 else 0.0
        wrist_score = 1.0 if wrist_middle_dist > 100 else 0.0
        palm_score = 1.0 if avg_palm_distance > 70 else 0.0
        
        # Weighted combination with palm distance
        open_score = (finger_score * 0.4 + thumb_score * 0.2 + wrist_score * 0.2 + palm_score * 0.2)
        
        return open_score
    
    def calculate_hand_size(self, landmarks, width, height):
        """Calculate hand size for adaptive thresholds."""
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        # Use wrist to middle finger as hand size reference
        hand_size = math.sqrt(
            (wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2
        ) * max(width, height)
        
        return hand_size
    
    def smooth_grip_detection(self, left_hand, right_hand):
        """Smooth grip detection using history."""
        if left_hand:
            self.left_grip_history.append(left_hand['grip_score'])
            if len(self.left_grip_history) > self.grip_history_size:
                self.left_grip_history.pop(0)
            
            # Use average of history
            avg_grip = sum(self.left_grip_history) / len(self.left_grip_history)
            self.left_hand_gripping = avg_grip > 0.3  # More sensitive
        
        if right_hand:
            self.right_grip_history.append(right_hand['grip_score'])
            if len(self.right_grip_history) > self.grip_history_size:
                self.right_grip_history.pop(0)
            
            # Use average of history
            avg_grip = sum(self.right_grip_history) / len(self.right_grip_history)
            self.right_hand_gripping = avg_grip > 0.3  # More sensitive
        
        # Both hands open for braking
        self.both_hands_open = (left_hand and left_hand['open_score'] > 0.4 and 
                              right_hand and right_hand['open_score'] > 0.4)
    
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
    
    def update_controls_robust(self, left_hand, right_hand):
        """Update controls with robust logic."""
        current_time = time.time()
        
        if current_time - self.last_key_time < self.key_delay:
            return
        
        # Update hand positions and states
        self.left_hand_pos = left_hand['center'] if left_hand else None
        self.right_hand_pos = right_hand['center'] if right_hand else None
        
        # Smooth grip detection
        self.smooth_grip_detection(left_hand, right_hand)
        
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
            
            # Draw grip status with scores
            grip_score = left_hand['grip_score']
            status = "GRIP" if self.left_hand_gripping else "OPEN" if left_hand['is_open'] else "NEUTRAL"
            
            cv2.putText(frame, f"L: {status} ({grip_score:.2f})", (center[0] - 50, center[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if right_hand:
            center = right_hand['center']
            color = self.colors['hand_right']
            cv2.circle(frame, center, 15, color, -1)
            cv2.circle(frame, center, 20, color, 3)
            
            # Draw grip status with scores
            grip_score = right_hand['grip_score']
            status = "GRIP" if self.right_hand_gripping else "OPEN" if right_hand['is_open'] else "NEUTRAL"
            
            cv2.putText(frame, f"R: {status} ({grip_score:.2f})", (center[0] - 50, center[1] - 30), 
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
        print("Robust Simple Racing Controller")
        print("===============================")
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
                left_hand, right_hand = self.detect_hands_robust(frame)
                
                # Update controls if calibrated
                if self.calibration_complete and left_hand and right_hand:
                    self.update_controls_robust(left_hand, right_hand)
                
                # Draw steering wheel
                self.draw_steering_wheel(frame)
                
                # Draw hands and line
                self.draw_hands_and_line(frame, left_hand, right_hand)
                
                # Draw controls
                self.draw_controls(frame)
                
                # Display the frame
                cv2.imshow('Robust Simple Racing Controller', frame)
                
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
    controller = RobustSimpleRacingController()
    controller.run()

if __name__ == "__main__":
    main()
