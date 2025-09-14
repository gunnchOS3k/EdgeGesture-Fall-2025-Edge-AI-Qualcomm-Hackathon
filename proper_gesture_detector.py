#!/usr/bin/env python3
"""
Proper Gesture Detector - Using correct MediaPipe landmarks and dimensions
Based on official MediaPipe documentation for 21 hand landmarks
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Dict, Tuple

class ProperGestureDetector:
    def __init__(self):
        """Initialize the proper gesture detector with correct MediaPipe setup."""
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Camera setup
        self.cap = None
        self.camera_active = False
        
        # Gesture detection state
        self.gesture_history = []
        self.history_size = 5
        
        # Colors for visualization
        self.colors = {
            'hand_left': (0, 255, 0),      # Green
            'hand_right': (0, 0, 255),     # Red
            'text': (255, 255, 255),       # White
            'gesture': (255, 255, 0),      # Yellow
            'success': (0, 255, 0),        # Green
            'warning': (0, 255, 255)       # Cyan
        }
    
    def start_camera(self, camera_index: int = 0) -> bool:
        """Start the camera with proper error handling."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print(f"❌ Error: Could not open camera {camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test camera
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error: Could not read from camera")
                return False
            
            self.camera_active = True
            print(f"✅ Camera {camera_index} started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera and release resources."""
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("📹 Camera stopped")
    
    def calculate_distance(self, point1, point2) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_extended(self, landmarks, finger_tip_idx: int, finger_pip_idx: int, finger_mcp_idx: int, hand_side: str) -> bool:
        """Proper finger extension check using correct MediaPipe landmarks."""
        tip = landmarks[finger_tip_idx]
        pip = landmarks[finger_pip_idx]
        mcp = landmarks[finger_mcp_idx]
        
        # For thumb (landmark 4), check based on hand orientation
        if finger_tip_idx == 4:  # Thumb tip
            if hand_side == "left":
                # Left hand: thumb extends to the right (tip.x > mcp.x)
                return tip.x > mcp.x and tip.x > pip.x
            else:
                # Right hand: thumb extends to the left (tip.x < mcp.x)
                return tip.x < mcp.x and tip.x < pip.x
        else:
            # For other fingers, check Y-coordinate (tip above pip means extended)
            return tip.y < pip.y
    
    def get_finger_states(self, landmarks, hand_side: str) -> Dict[str, bool]:
        """Get finger states using correct MediaPipe landmark indices."""
        finger_states = {}
        
        # Thumb (landmarks: CMC=1, MCP=2, IP=3, Tip=4)
        finger_states['thumb'] = self.is_finger_extended(landmarks, 4, 3, 2, hand_side)
        
        # Index finger (landmarks: MCP=5, PIP=6, DIP=7, Tip=8)
        finger_states['index'] = self.is_finger_extended(landmarks, 8, 6, 5, hand_side)
        
        # Middle finger (landmarks: MCP=9, PIP=10, DIP=11, Tip=12)
        finger_states['middle'] = self.is_finger_extended(landmarks, 12, 10, 9, hand_side)
        
        # Ring finger (landmarks: MCP=13, PIP=14, DIP=15, Tip=16)
        finger_states['ring'] = self.is_finger_extended(landmarks, 16, 14, 13, hand_side)
        
        # Pinky (landmarks: MCP=17, PIP=18, DIP=19, Tip=20)
        finger_states['pinky'] = self.is_finger_extended(landmarks, 20, 18, 17, hand_side)
        
        return finger_states
    
    def detect_gesture(self, landmarks, hand_side: str) -> Tuple[str, float]:
        """Proper gesture detection using correct MediaPipe landmarks."""
        if not landmarks:
            return "none", 0.0
        
        # Get finger states with proper hand orientation
        finger_states = self.get_finger_states(landmarks, hand_side)
        
        # Count extended fingers
        extended_count = sum(finger_states.values())
        
        # Get specific finger states
        thumb_extended = finger_states['thumb']
        index_extended = finger_states['index']
        middle_extended = finger_states['middle']
        ring_extended = finger_states['ring']
        pinky_extended = finger_states['pinky']
        
        # Calculate distances for validation
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        
        # 1. FIST - All fingers closed (including thumb)
        if extended_count == 0:
            return "fist", 0.9
        
        # 2. THUMBS UP - Only thumb extended, others closed
        if (thumb_extended and 
            not index_extended and 
            not middle_extended and 
            not ring_extended and 
            not pinky_extended and
            thumb_index_dist > 0.1):  # Thumb should be away from other fingers
            return "thumbs_up", 0.9
        
        # 3. PEACE SIGN - Only index and middle extended
        if (not thumb_extended and 
            index_extended and 
            middle_extended and 
            not ring_extended and 
            not pinky_extended):
            return "peace", 0.9
        
        # 4. OK SIGN - Thumb and index extended, others closed
        if (thumb_extended and 
            index_extended and 
            not middle_extended and 
            not ring_extended and 
            not pinky_extended and
            0.05 < thumb_index_dist < 0.15):  # Close but not touching
            return "ok", 0.9
        
        # 5. POINT - Only index finger extended
        if (not thumb_extended and 
            index_extended and 
            not middle_extended and 
            not ring_extended and 
            not pinky_extended):
            return "point", 0.9
        
        # 6. FINGER COUNTING - Based on number of extended fingers
        if extended_count == 1:
            if thumb_extended:
                return "thumbs_up", 0.8
            else:
                return "one_finger", 0.8
        elif extended_count == 2:
            if index_extended and middle_extended:
                return "peace", 0.8
            elif thumb_extended and index_extended:
                return "ok", 0.8
            else:
                return "two_fingers", 0.8
        elif extended_count == 3:
            return "three_fingers", 0.8
        elif extended_count == 4:
            # CRITICAL: 4 fingers means pinky is NOT extended
            if not pinky_extended:
                return "four_fingers", 0.9
            else:
                return "four_fingers", 0.7
        elif extended_count == 5:
            # OPEN HAND - All fingers extended
            return "open_hand", 0.9
        
        return "none", 0.0
    
    def get_stable_gesture(self, gestures: List[Dict]) -> str:
        """Get the most stable gesture from recent history."""
        if not gestures:
            return "none"
        
        # Count gestures in recent history
        gesture_counts = {}
        for frame_gestures in self.gesture_history[-3:]:  # Last 3 frames
            for hand_data in frame_gestures:
                key = f"{hand_data['side']}_{hand_data['gesture']}"
                gesture_counts[key] = gesture_counts.get(key, 0) + 1
        
        # Return most common gesture
        if gesture_counts:
            most_common = max(gesture_counts, key=gesture_counts.get)
            side, gesture = most_common.split('_', 1)
            return gesture
        
        return "none"
    
    def process_frame(self, frame) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame for gesture detection."""
        if not self.camera_active or frame is None:
            return frame, []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        detected_gestures = []
        
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Determine hand side using MediaPipe's handedness
                if results.multi_handedness:
                    hand_side = results.multi_handedness[idx].classification[0].label.lower()
                else:
                    # Fallback: use landmark position
                    hand_center_x = hand_landmarks.landmark[9].x  # Middle finger MCP
                    hand_side = "left" if hand_center_x < 0.5 else "right"
                
                # Detect gesture with proper hand orientation
                gesture, confidence = self.detect_gesture(hand_landmarks.landmark, hand_side)
                
                # Store hand data
                hand_data = {
                    'side': hand_side,
                    'gesture': gesture,
                    'confidence': confidence,
                    'landmarks': hand_landmarks.landmark,
                    'center': (int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h))
                }
                
                detected_gestures.append(hand_data)
                
                # Draw gesture info
                center = hand_data['center']
                cv2.circle(frame, center, 12, self.colors[f'hand_{hand_side}'], -1)
                cv2.circle(frame, center, 15, self.colors[f'hand_{hand_side}'], 2)
                
                # Draw gesture text
                text = f"{hand_side.upper()}: {gesture.upper()}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = center[0] - text_size[0] // 2
                text_y = center[1] - 25
                
                # Draw background
                cv2.rectangle(frame, 
                            (text_x - 5, text_y - text_size[1] - 5), 
                            (text_x + text_size[0] + 5, text_y + 5), 
                            (0, 0, 0), -1)
                
                cv2.putText(frame, text, 
                           (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['gesture'], 2)
                
                # Draw confidence
                conf_text = f"({confidence:.2f})"
                conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                conf_x = center[0] - conf_size[0] // 2
                conf_y = center[1] + 20
                
                cv2.rectangle(frame, 
                            (conf_x - 3, conf_y - conf_size[1] - 3), 
                            (conf_x + conf_size[0] + 3, conf_y + 3), 
                            (0, 0, 0), -1)
                
                cv2.putText(frame, conf_text, 
                           (conf_x, conf_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gesture'], 1)
        
        # Update gesture history
        self.gesture_history.append(detected_gestures)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        return frame, detected_gestures
    
    def run_demo(self):
        """Run a demo of the proper gesture detector."""
        print("🎮 Proper Gesture Detector")
        print("==========================")
        print("Using correct MediaPipe landmarks and dimensions")
        print("Press 'q' to quit, 'r' to reset history")
        print()
        
        if not self.start_camera():
            return
        
        try:
            while self.camera_active:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, gestures = self.process_frame(frame)
                
                # Draw status
                cv2.putText(processed_frame, f"Gestures: {len(gestures)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
                
                # Draw stable gesture
                stable_gesture = self.get_stable_gesture(gestures)
                cv2.putText(processed_frame, f"Stable: {stable_gesture.upper()}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['success'], 2)
                
                # Draw instructions
                instructions = [
                    "Proper Gesture Detection:",
                    "• Fist (all fingers down)",
                    "• Open Hand (all fingers up)",
                    "• Thumbs Up (only thumb up)",
                    "• Peace (index + middle up)",
                    "• OK (thumb + index circle)",
                    "• Point (only index up)",
                    "• Count 1-5 fingers"
                ]
                
                y_offset = 90
                for instruction in instructions:
                    cv2.putText(processed_frame, instruction, 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                    y_offset += 20
                
                # Display the frame
                cv2.imshow('Proper Gesture Detector', processed_frame)
                
                # Check for commands
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.gesture_history.clear()
                    print("🔄 Gesture history reset")
        
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        self.stop_camera()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def main():
    """Main entry point for testing."""
    detector = ProperGestureDetector()
    detector.run_demo()

if __name__ == "__main__":
    main()
