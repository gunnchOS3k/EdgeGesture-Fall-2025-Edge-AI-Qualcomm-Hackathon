#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed.
Run this before running the main application.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        print(f"  OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ Mediapipe imported successfully")
        print(f"  Mediapipe version: {mp.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Mediapipe: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import pyautogui
        print("✓ pyautogui imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pyautogui: {e}")
        return False
    
    return True

def test_camera():
    """Test if camera can be accessed."""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera is accessible and working")
                print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("✗ Camera opened but cannot read frames")
                cap.release()
                return False
            cap.release()
        else:
            print("✗ Cannot open camera (might be in use by another application)")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False
    
    return True

def test_hand_detector():
    """Test if HandDetector can be initialized."""
    print("\nTesting HandDetector initialization...")
    
    try:
        from hand_detector import HandDetector
        detector = HandDetector()
        print("✓ HandDetector initialized successfully")
        return True
    except Exception as e:
        print(f"✗ HandDetector initialization failed: {e}")
        return False

def test_gesture_controller():
    """Test if GestureController can be initialized."""
    print("\nTesting GestureController initialization...")
    
    try:
        from gesture_controller import GestureController
        controller = GestureController()
        print("✓ GestureController initialized successfully")
        controller.cleanup()
        return True
    except Exception as e:
        print(f"✗ GestureController initialization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Gesture-to-Keyboard Controller - Installation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test camera
    if not test_camera():
        all_tests_passed = False
    
    # Test custom modules
    if not test_hand_detector():
        all_tests_passed = False
    
    if not test_gesture_controller():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed! You're ready to run the application.")
        print("\nTo start the application, run:")
        print("  python main.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("\nTo install dependencies, run:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
