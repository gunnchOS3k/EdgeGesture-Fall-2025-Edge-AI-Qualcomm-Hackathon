#!/usr/bin/env python3
"""
Run script for the Gesture-to-Keyboard Controller.
This script helps with camera permissions and runs the main application.
"""

import subprocess
import sys
import os

def check_camera_permissions():
    """Check if camera permissions are granted."""
    print("Checking camera permissions...")
    
    # Try to import and test camera access
    try:
        import cv2
        cap = cv2.VideoCapture(0) 
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Camera permissions are granted!")
                return True
            else:
                print("✗ Camera opened but cannot read frames")
                return False
        else:
            print("✗ Camera access denied")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def request_camera_permissions():
    """Request camera permissions by opening System Preferences."""
    print("\n" + "="*60)
    print("CAMERA PERMISSION REQUIRED")
    print("="*60)
    print("To use the gesture controller, you need to grant camera access.")
    print("\nFollow these steps:")
    print("1. A dialog should appear asking for camera permission")
    print("2. If not, go to: System Preferences > Security & Privacy > Privacy > Camera")
    print("3. Find 'Terminal' or 'Python' in the list and check the box")
    print("4. Restart the application")
    print("\nAlternatively, you can run this command to open System Preferences:")
    print("open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Camera'")
    print("="*60)
    
    # Try to open System Preferences
    try:
        subprocess.run([
            "open", 
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
        ], check=False)
        print("\nOpening System Preferences...")
    except Exception as e:
        print(f"Could not open System Preferences automatically: {e}")

def main():
    """Main function to run the gesture controller."""
    print("Gesture-to-Keyboard Controller Launcher")
    print("="*40)
    
    # Check if we're using the right Python
    python_path = "/opt/homebrew/bin/python3.11"
    if not os.path.exists(python_path):
        print("Error: Python 3.11 not found at expected location")
        print("Please make sure Python 3.11 is installed via Homebrew")
        return
    
    # Check camera permissions
    if not check_camera_permissions():
        request_camera_permissions()
        print("\nAfter granting permissions, run this script again.")
        return
    
    # Run the main application
    print("\nStarting gesture controller...")
    print("Press 'q' in the camera window to quit")
    print("Make sure you have a game open to control!")
    print("-" * 40)
    
    try:
        subprocess.run([python_path, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"Application failed with error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
