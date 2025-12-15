## 🎯 Features

- **Real-time Gesture Recognition**: Detects multiple hand gestures using MediaPipe hand tracking
- **Dynamic Meme Overlay**: Smoothly overlays meme images based on detected gestures
- **Multiple Gesture Support**: 
  - 🤔 Thinking pose (hand near chin)
  - ☝️ Pointing up (index finger extended)
  - 😮 Surprised (hands near face, fingers spread)
  - 😐 Neutral (default state)
- **Extensible Architecture**: Easy to add new gestures and memes

## 📋 Requirements

- Python 3.8 or higher
- Webcam
- Operating System: Windows, macOS, or Linux

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/hacminhquan/gesture-meme-overlay.git
cd gesture-meme-overlay
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

```
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.21.0
pillow>=9.0.0
```

## 📁 Project Structure

```
gesture-meme-overlay/
│
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
│
├── memes/                 # Meme images directory
│   ├── monkey_thinking.jpg
│   ├── monkey_pointing.jpg
│   ├── monkey_neutral.jpg
│   └── monkey_surprised.jpg
│
├── docs/                  # Additional documentation
│   ├── ARCHITECTURE.md
│   └── CONTRIBUTING.md
│
└── tests/                 # Unit tests
    └── test_gestures.py
```

## 🎮 Usage

### Basic Usage

```bash
python main.py
```

### Controls

- **Q**: Quit the application
- The camera feed will mirror your movements for intuitive interaction

### Adding Custom Memes

1. Place your meme images in the `memes/` directory
2. Name them according to the gesture:
   - `monkey_thinking.jpg`
   - `monkey_pointing.jpg`
   - `monkey_neutral.jpg`
   - `monkey_surprised.jpg`
3. The application will automatically load them on startup

## 🏗️ Architecture

### Core Components

1. **GestureRecognizer**: Processes MediaPipe hand landmarks and identifies gestures
2. **MemeOverlay**: Handles image overlay with smooth fade transitions
3. **GestureMemeApp**: Main application orchestrator
4. **FPSCounter**: Tracks and displays performance metrics

### Gesture Detection Algorithm

The system uses geometric analysis of hand landmarks:

- **Distance calculations** between key points
- **Finger extension detection** based on joint positions
- **Position analysis** relative to frame regions
- **Multi-criteria validation** for robust recognition

## 🎨 Customization

### Adjust Gesture Sensitivity

Modify the `GestureConfig` class in `main.py`:

```python
@dataclass
class GestureConfig:
    thinking_thumb_to_chin_ratio: float = 0.15
    pointing_finger_extension_threshold: float = 0.6
    surprised_mouth_open_threshold: float = 0.08
    confidence_threshold: float = 0.7
```

### Change Overlay Position/Size

Modify the `MemeOverlay` initialization:

```python
self.overlay_size = (200, 200)  # Width, Height in pixels
self.overlay_position = (20, 20)  # X, Y coordinates
```

### Add New Gestures

1. Implement detection logic in `GestureRecognizer.recognize_gesture()`
2. Add corresponding meme image to `memes/` directory
3. Update `MemeOverlay.load_memes()` with new gesture mapping

## 📊 Performance

- **FPS**: 30+ on modern hardware
- **Latency**: <50ms gesture detection
- **CPU Usage**: ~15-25% on Intel i5/i7

## 🐛 Troubleshooting

### Camera not detected
- Ensure no other application is using the camera
- Check camera permissions in system settings
- Try changing camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

### Low FPS
- Close other resource-intensive applications
- Reduce camera resolution in `initialize()` method
- Disable hand landmark drawing for better performance

### Gesture not recognized
- Ensure good lighting conditions
- Position hand clearly in front of camera
- Adjust sensitivity thresholds in `GestureConfig`

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for hand tracking
- [OpenCV](https://opencv.org/) for computer vision capabilities
- Monkey meme images from [source]


Project Link: [https://github.com/hacminhquan/gesture-meme-overlay](https://github.com/hacminhquan/gesture-meme-overlay)

## 🗺️ Roadmap

- [ ] Add more gesture types (peace sign, thumbs up, etc.)
- [ ] Support for multiple simultaneous overlays
- [ ] Custom gesture training interface
- [ ] Video recording with overlay
- [ ] Mobile app version
- [ ] Real-time gesture statistics
- [ ] Cloud-based gesture model training

## 📚 Related Projects

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Face Detection](https://github.com/opencv/opencv)
- [Hand Gesture Recognition Projects](https://github.com/topics/hand-gesture-recognition)

---

⭐ If you found this project useful, please consider giving it a star!
