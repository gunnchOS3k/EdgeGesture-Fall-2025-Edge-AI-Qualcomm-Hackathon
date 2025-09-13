#!/usr/bin/env python3
"""
Advanced Racing Controller with Robust Hand Detection
Uses multiple computer vision techniques for accurate grip detection
"""

import cv2
import numpy as np
import math
import pyautogui
import time
from hand_detector import HandDetector

class AdvancedRacingController:
    def __init__(self):
        """Initialize the advanced racing controller."""
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
        
        # Hand tracking with advanced smoothing
        self.left_hand_pos = None
        self.right_hand_pos = None
        self.left_hand_gripping = False
        self.right_hand_gripping = False
        self.both_hands_open = False
        
        # Advanced grip detection
        self.grip_confidence_threshold = 0.7
        self.open_confidence_threshold = 0.7
        
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
    
    def detect_hands_advanced(self, frame):
        """Advanced hand detection with multiple techniques."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hand_detector.hands.process(rgb_frame)
        
        left_hand_data = None
        right_hand_data = None
        
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand data with advanced analysis
                hand_data = self.extract_hand_data_advanced(hand_landmarks, w, h)
                
                # Determine hand side based on position relative to center
                if hand_data['center'][0] < w // 2:
                    left_hand_data = hand_data
                else:
                    right_hand_data = hand_data
        
        return left_hand_data, right_hand_data
    
    def extract_hand_data_advanced(self, hand_landmarks, width, height):
        """Extract hand data with advanced grip detection."""
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
        
        # Advanced grip detection using multiple methods
        grip_analysis = self.analyze_grip_advanced(hand_landmarks, width, height)
        open_analysis = self.analyze_open_advanced(hand_landmarks, width, height)
        
        # Determine hand state based on analysis
        is_gripping = grip_analysis['confidence'] > self.grip_confidence_threshold
        is_open = open_analysis['confidence'] > self.open_confidence_threshold
        
        return {
            'center': hand_center,
            'is_gripping': is_gripping,
            'is_open': is_open,
            'grip_analysis': grip_analysis,
            'open_analysis': open_analysis
        }
    
    def analyze_grip_advanced(self, hand_landmarks, width, height):
        """Advanced grip analysis using multiple computer vision techniques."""
        landmarks = hand_landmarks.landmark
        
        # Method 1: Finger tip to PIP distance analysis
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        closed_fingers = 0
        finger_distances = []
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_pos = (landmarks[tip_idx].x * width, landmarks[tip_idx].y * height)
            pip_pos = (landmarks[pip_idx].x * width, landmarks[pip_idx].y * height)
            
            distance = math.sqrt((tip_pos[0] - pip_pos[0])**2 + (tip_pos[1] - pip_pos[1])**2)
            finger_distances.append(distance)
            
            # More sophisticated threshold based on hand size
            hand_size = self.calculate_hand_size(landmarks, width, height)
            threshold = hand_size * 0.15  # Adaptive threshold
            
            if distance < threshold:
                closed_fingers += 1
        
        # Method 2: Thumb opposition analysis
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        
        # Calculate thumb opposition
        thumb_opposition = self.calculate_thumb_opposition(thumb_tip, thumb_ip, index_tip, index_mcp, width, height)
        
        # Method 3: Hand convexity analysis
        hand_convexity = self.calculate_hand_convexity(landmarks, width, height)
        
        # Method 4: Finger curvature analysis
        finger_curvature = self.calculate_finger_curvature(landmarks, width, height)
        
        # Calculate confidence scores
        finger_score = closed_fingers / 5.0
        thumb_score = 1.0 if thumb_opposition > 0.7 else 0.0
        convexity_score = 1.0 if hand_convexity > 0.8 else 0.0
        curvature_score = 1.0 if finger_curvature > 0.6 else 0.0
        
        # Weighted combination with more sophisticated weighting
        confidence = (
            finger_score * 0.4 +
            thumb_score * 0.3 +
            convexity_score * 0.2 +
            curvature_score * 0.1
        )
        
        return {
            'confidence': confidence,
            'finger_score': finger_score,
            'thumb_score': thumb_score,
            'convexity_score': convexity_score,
            'curvature_score': curvature_score,
            'closed_fingers': closed_fingers,
            'finger_distances': finger_distances
        }
    
    def analyze_open_advanced(self, hand_landmarks, width, height):
        """Advanced open palm analysis."""
        landmarks = hand_landmarks.landmark
        
        # Method 1: Finger spread analysis
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        open_fingers = 0
        finger_distances = []
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_pos = (landmarks[tip_idx].x * width, landmarks[tip_idx].y * height)
            pip_pos = (landmarks[pip_idx].x * width, landmarks[pip_idx].y * height)
            
            distance = math.sqrt((tip_pos[0] - pip_pos[0])**2 + (tip_pos[1] - pip_pos[1])**2)
            finger_distances.append(distance)
            
            # Adaptive threshold for open detection
            hand_size = self.calculate_hand_size(landmarks, width, height)
            threshold = hand_size * 0.25  # Higher threshold for open detection
            
            if distance > threshold:
                open_fingers += 1
        
        # Method 2: Palm area analysis
        palm_area = self.calculate_palm_area(landmarks, width, height)
        
        # Method 3: Finger spread angle analysis
        spread_angle = self.calculate_finger_spread(landmarks, width, height)
        
        # Calculate confidence scores
        finger_score = open_fingers / 5.0
        area_score = min(1.0, palm_area / 10000)  # Normalize area
        spread_score = 1.0 if spread_angle > 0.7 else 0.0
        
        # Weighted combination
        confidence = (
            finger_score * 0.5 +
            area_score * 0.3 +
            spread_score * 0.2
        )
        
        return {
            'confidence': confidence,
            'finger_score': finger_score,
            'area_score': area_score,
            'spread_score': spread_score,
            'open_fingers': open_fingers,
            'palm_area': palm_area
        }
    
    def calculate_hand_size(self, landmarks, width, height):
        """Calculate hand size for adaptive thresholds."""
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        # Use wrist to middle finger as hand size reference
        hand_size = math.sqrt(
            (wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2
        ) * max(width, height)
        
        return hand_size
    
    def calculate_thumb_opposition(self, thumb_tip, thumb_ip, index_tip, index_mcp, width, height):
        """Calculate thumb opposition (how well thumb opposes other fingers)."""
        # Calculate vectors
        thumb_vec = np.array([thumb_tip.x - thumb_ip.x, thumb_tip.y - thumb_ip.y])
        index_vec = np.array([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y])
        
        # Normalize vectors
        thumb_vec = thumb_vec / (np.linalg.norm(thumb_vec) + 1e-8)
        index_vec = index_vec / (np.linalg.norm(index_vec) + 1e-8)
        
        # Calculate opposition (dot product)
        opposition = np.dot(thumb_vec, index_vec)
        
        return opposition
    
    def calculate_hand_convexity(self, landmarks, width, height):
        """Calculate hand convexity (closed hand is more convex)."""
        # Get hand contour points
        hand_points = []
        for landmark in landmarks:
            hand_points.append([landmark.x * width, landmark.y * height])
        
        hand_points = np.array(hand_points, dtype=np.int32)
        
        # Calculate convex hull
        hull = cv2.convexHull(hand_points)
        
        # Calculate convexity defects
        defects = cv2.convexityDefects(hand_points, hull)
        
        if defects is not None:
            # Count significant defects (fingers create valleys)
            significant_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 1000:  # Threshold for significant defect
                    significant_defects += 1
            
            # More defects = more open hand
            convexity = 1.0 - min(1.0, significant_defects / 5.0)
        else:
            convexity = 0.5
        
        return convexity
    
    def calculate_finger_curvature(self, landmarks, width, height):
        """Calculate finger curvature (closed fingers are more curved)."""
        # Analyze each finger's curvature
        finger_landmarks = [
            [4, 3, 2],    # Thumb
            [8, 7, 6],    # Index
            [12, 11, 10], # Middle
            [16, 15, 14], # Ring
            [20, 19, 18]  # Pinky
        ]
        
        total_curvature = 0
        valid_fingers = 0
        
        for finger in finger_landmarks:
            if len(finger) >= 3:
                p1 = landmarks[finger[0]]
                p2 = landmarks[finger[1]]
                p3 = landmarks[finger[2]]
                
                # Calculate curvature using cross product
                v1 = np.array([p2.x - p1.x, p2.y - p1.y])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y])
                
                cross_product = np.cross(v1, v2)
                curvature = abs(cross_product)
                
                total_curvature += curvature
                valid_fingers += 1
        
        if valid_fingers > 0:
            avg_curvature = total_curvature / valid_fingers
            # Normalize curvature
            normalized_curvature = min(1.0, avg_curvature / 0.1)
        else:
            normalized_curvature = 0.0
        
        return normalized_curvature
    
    def calculate_palm_area(self, landmarks, width, height):
        """Calculate palm area."""
        # Use palm landmarks to estimate area
        palm_landmarks = [0, 1, 5, 9, 13, 17]  # Wrist and finger bases
        
        palm_points = []
        for idx in palm_landmarks:
            landmark = landmarks[idx]
            palm_points.append([landmark.x * width, landmark.y * height])
        
        palm_points = np.array(palm_points, dtype=np.int32)
        
        # Calculate area using shoelace formula
        area = 0.5 * abs(sum(palm_points[i][0] * palm_points[(i+1) % len(palm_points)][1] - 
                            palm_points[(i+1) % len(palm_points)][0] * palm_points[i][1] 
                            for i in range(len(palm_points))))
        
        return area
    
    def calculate_finger_spread(self, landmarks, width, height):
        """Calculate finger spread angle."""
        # Get finger tip positions
        finger_tips = [4, 8, 12, 16, 20]
        
        tip_positions = []
        for idx in finger_tips:
            landmark = landmarks[idx]
            tip_positions.append([landmark.x * width, landmark.y * height])
        
        tip_positions = np.array(tip_positions)
        
        # Calculate spread angle
        center = np.mean(tip_positions, axis=0)
        
        angles = []
        for i in range(len(tip_positions)):
            for j in range(i+1, len(tip_positions)):
                v1 = tip_positions[i] - center
                v2 = tip_positions[j] - center
                
                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = math.acos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        if angles:
            avg_angle = np.mean(angles)
            # Normalize to 0-1 (wider spread = higher score)
            spread_score = min(1.0, avg_angle / (math.pi / 2))
        else:
            spread_score = 0.0
        
        return spread_score
    
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
    
    def update_controls_advanced(self, left_hand, right_hand):
        """Update controls with advanced logic."""
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
        """Draw hands and the line between them with detailed analysis."""
        # Draw hands
        if left_hand:
            center = left_hand['center']
            color = self.colors['hand_left']
            cv2.circle(frame, center, 15, color, -1)
            cv2.circle(frame, center, 20, color, 3)
            
            # Draw detailed analysis
            grip_analysis = left_hand['grip_analysis']
            status = "GRIP" if self.left_hand_gripping else "OPEN" if left_hand['is_open'] else "NEUTRAL"
            
            cv2.putText(frame, f"L: {status}", (center[0] - 50, center[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Conf: {grip_analysis['confidence']:.2f}", (center[0] - 50, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if right_hand:
            center = right_hand['center']
            color = self.colors['hand_right']
            cv2.circle(frame, center, 15, color, -1)
            cv2.circle(frame, center, 20, color, 3)
            
            # Draw detailed analysis
            grip_analysis = right_hand['grip_analysis']
            status = "GRIP" if self.right_hand_gripping else "OPEN" if right_hand['is_open'] else "NEUTRAL"
            
            cv2.putText(frame, f"R: {status}", (center[0] - 50, center[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Conf: {grip_analysis['confidence']:.2f}", (center[0] - 50, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
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
        """Draw control status with detailed information."""
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
        print("Advanced Racing Controller")
        print("=========================")
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
                left_hand, right_hand = self.detect_hands_advanced(frame)
                
                # Update controls if calibrated
                if self.calibration_complete and left_hand and right_hand:
                    self.update_controls_advanced(left_hand, right_hand)
                
                # Draw steering wheel
                self.draw_steering_wheel(frame)
                
                # Draw hands and line
                self.draw_hands_and_line(frame, left_hand, right_hand)
                
                # Draw controls
                self.draw_controls(frame)
                
                # Display the frame
                cv2.imshow('Advanced Racing Controller', frame)
                
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
    controller = AdvancedRacingController()
    controller.run()

if __name__ == "__main__":
    main()
