import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List

class HandDetector:
    def __init__(self):
        """Initialize Mediapipe hands solution."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only detect one hand for simplicity
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_hand(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], bool, Optional[np.ndarray]]:
        """
        Detect hand in the frame and return index finger position and gesture info.
        
        Args:
            frame: Input BGR frame from webcam
            
        Returns:
            Tuple of:
            - index_finger_pos: (x, y) coordinates of index finger tip, or None if not detected
            - is_fist: Boolean indicating if hand is closed (fist gesture)
            - annotated_frame: Frame with hand landmarks drawn (for debugging)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        index_finger_pos = None
        is_fist = False
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            # Get the first (and only) hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks on the frame
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Get index finger tip (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            index_finger_pos = (int(index_tip.x * w), int(index_tip.y * h))
            
            # Check if hand is closed (fist gesture)
            # We'll check if fingertips are close to their respective knuckles
            is_fist = self._detect_fist(hand_landmarks, w, h)
            
        return index_finger_pos, is_fist, annotated_frame
    
    def _detect_fist(self, hand_landmarks, width: int, height: int) -> bool:
        """
        Detect if the hand is closed into a fist.
        
        Args:
            hand_landmarks: Mediapipe hand landmarks
            width: Frame width
            height: Frame height
            
        Returns:
            True if hand is detected as a fist, False otherwise
        """
        # Get key landmarks
        landmarks = hand_landmarks.landmark
        
        # Convert to pixel coordinates
        def to_pixel(landmark):
            return (int(landmark.x * width), int(landmark.y * height))
        
        # Check if fingertips are close to their respective knuckles
        # This is a simple heuristic - in a fist, fingertips should be close to knuckles
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        closed_fingers = 0
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_pos = to_pixel(landmarks[tip_idx])
            pip_pos = to_pixel(landmarks[pip_idx])
            
            # Calculate distance between tip and PIP
            distance = np.sqrt((tip_pos[0] - pip_pos[0])**2 + (tip_pos[1] - pip_pos[1])**2)
            
            # If distance is small, finger is likely closed
            if distance < 30:  # Threshold in pixels
                closed_fingers += 1
        
        # Consider it a fist if 4 or more fingers are closed
        return closed_fingers >= 4
    
    def get_screen_region(self, x_pos: int, frame_width: int) -> str:
        """
        Determine which third of the screen the x position is in.
        
        Args:
            x_pos: X coordinate of the finger
            frame_width: Width of the frame
            
        Returns:
            'left', 'middle', or 'right'
        """
        third = frame_width // 3
        
        if x_pos < third:
            return 'left'
        elif x_pos < 2 * third:
            return 'middle'
        else:
            return 'right'
