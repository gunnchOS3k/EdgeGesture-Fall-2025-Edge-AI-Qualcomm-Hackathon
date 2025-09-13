import pyautogui
import time
from typing import Optional

class GestureController:
    def __init__(self):
        """Initialize the gesture controller for keyboard simulation."""
        # Track currently pressed keys to avoid redundant key presses
        self.current_keys = set()
        
        # Key mappings
        self.LEFT_KEY = 'left'
        self.RIGHT_KEY = 'right'
        self.SPACE_KEY = 'space'
        
        # Small delay to prevent too rapid key presses
        self.last_key_time = 0
        self.key_delay = 0.05  # 50ms delay between key operations
        
    def handle_gesture(self, screen_region: str, is_fist: bool) -> None:
        """
        Handle gesture input and send appropriate keyboard commands.
        
        Args:
            screen_region: 'left', 'middle', or 'right' - which third of screen finger is in
            is_fist: Whether hand is closed (fist gesture)
        """
        current_time = time.time()
        
        # Only process if enough time has passed since last key operation
        if current_time - self.last_key_time < self.key_delay:
            return
            
        # Determine which keys should be pressed based on gesture
        target_keys = set()
        
        if screen_region == 'left':
            target_keys.add(self.LEFT_KEY)
        elif screen_region == 'right':
            target_keys.add(self.RIGHT_KEY)
        # For 'middle', we don't add any movement keys
        
        # Add space key if fist is detected
        if is_fist:
            target_keys.add(self.SPACE_KEY)
        
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
        self.last_key_time = current_time
    
    def release_all_keys(self) -> None:
        """Release all currently pressed keys."""
        for key in self.current_keys.copy():
            try:
                pyautogui.keyUp(key)
                print(f"Released key: {key}")
            except Exception as e:
                print(f"Error releasing key {key}: {e}")
        
        self.current_keys.clear()
    
    def cleanup(self) -> None:
        """Clean up resources and release all keys."""
        self.release_all_keys()
        print("Gesture controller cleaned up - all keys released")
