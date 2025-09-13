# 🏎️ Racing Game with Gesture Control

A 3D-style racing game controlled entirely through hand gestures! Features a virtual steering wheel that maps your hand rotation to WASD controls.

## 🎮 Features

### Virtual Steering Wheel
- **On-screen steering wheel** that rotates based on your hand angle
- **Smooth WASD mapping** - no individual key presses needed
- **Real-time angle feedback** showing your steering input

### Gesture Controls
- **Hand Rotation**: Rotate your hand left/right to steer the car
- **Open Hand**: Accelerate (W key)
- **Fist Gesture**: Brake (S key)
- **Automatic A/D**: Left/right steering automatically controls A/D keys

### Visual Design
- **3D perspective road** with realistic proportions
- **Professional HUD** showing speed and control status
- **Smooth animations** and responsive controls
- **Real-time gesture feedback** in camera window

## 🚀 Quick Start

### Option 1: Racing Game Only (Recommended)
```bash
python racing_launcher.py
```

### Option 2: Test Without Camera
```bash
python test_racing.py
```
Use keyboard controls: W (accelerate), S (brake), A/D (steer)

### Option 3: Multi-Game Mode
```bash
python main.py
```
Press 'r' for racing game, 'p' for pong game

## 🎯 How to Play

1. **Start the game** using one of the methods above
2. **Position your hand** in front of the camera
3. **Rotate your hand** left/right to steer the car
4. **Open your hand** to accelerate
5. **Make a fist** to brake
6. **Watch the steering wheel** on screen rotate with your hand!

## 🔧 Technical Details

### Gesture Detection
- Uses MediaPipe for hand landmark detection
- Calculates hand angle from wrist to middle finger base
- Maps angle range (-180° to 180°) to steering range (-45° to 45°)

### Control Mapping
- **Hand Angle < -5°**: Press A key (turn left)
- **Hand Angle > 5°**: Press D key (turn right)
- **Open Hand**: Press W key (accelerate)
- **Fist Gesture**: Press S key (brake)

### Game Architecture
- Modular design with base game class
- Separate camera thread for gesture processing
- Real-time gesture data streaming to game

## 🎨 Visual Elements

### Racing Game
- 3D perspective road with center line
- Red racing car with direction indicator
- Speedometer and control status display
- Professional color scheme

### Camera Overlay
- Hand landmark visualization
- Real-time angle display
- Control status indicators
- Game-specific instructions

## 🛠️ Requirements

- Python 3.11+
- OpenCV
- MediaPipe
- Pygame
- PyAutoGUI
- NumPy

Install with:
```bash
pip install -r requirements.txt
```

## 🎮 Controls Summary

| Gesture | Action | Key Pressed |
|---------|--------|-------------|
| Rotate hand left | Steer left | A |
| Rotate hand right | Steer right | D |
| Open hand | Accelerate | W |
| Make fist | Brake | S |
| Press 'q' | Quit game | - |
| Press 'r' | Switch to racing | - |
| Press 'p' | Switch to pong | - |

## 🚧 Future Enhancements

- [ ] Multiple car models
- [ ] Track selection
- [ ] AI opponents
- [ ] Power-ups and obstacles
- [ ] Sound effects
- [ ] Lap timing
- [ ] High score system

## 🐛 Troubleshooting

### Camera Issues
- Ensure camera permissions are granted
- Check camera is not being used by other applications
- Try different camera indices (0, 1, 2)

### Gesture Detection
- Ensure good lighting
- Keep hand in camera view
- Avoid cluttered backgrounds
- Make sure hand is clearly visible

### Game Performance
- Close other applications for better performance
- Ensure stable camera connection
- Check system requirements

## 📝 Notes

- The steering wheel provides intuitive control similar to real racing games
- Gesture sensitivity can be adjusted in the code
- The game runs at 60 FPS for smooth gameplay
- Camera runs at 30 FPS for responsive gesture detection

Enjoy racing with gesture control! 🏁
