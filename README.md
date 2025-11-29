# Finger Counter

A real-time hand gesture recognition system that detects and counts how many fingers you're holding up using your webcam.

## Features

- **Real-time Detection**: Instantly recognizes hand gestures through your webcam
- **Accurate Finger Counting**: Counts 0-5 fingers per hand
- **Multi-hand Support**: Can detect and count fingers on both hands simultaneously
- **Visual Feedback**: Displays hand landmarks and finger count on screen
- **FPS Counter**: Shows real-time performance metrics

## Requirements

- Python 3.7 or higher
- Webcam

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the finger counter:

```bash
python finger_counter.py
```

### Controls

- **q**: Quit the application

## How It Works

The application uses:
- **MediaPipe**: Google's ML framework for hand landmark detection
- **OpenCV**: For webcam capture and image processing

The system detects 21 hand landmarks and analyzes finger positions to determine which fingers are extended.

## Tips for Best Results

- Ensure good lighting conditions
- Keep your hand within the camera frame
- Hold fingers clearly extended or folded for accurate counting
- The image is mirrored for natural interaction

## Troubleshooting

**Webcam not opening:**
- Check if another application is using the webcam
- Verify webcam permissions in system settings

**Low FPS:**
- Close other resource-intensive applications
- Reduce `max_num_hands` in the code if only detecting one hand

**Inaccurate counting:**
- Adjust `min_detection_confidence` and `min_tracking_confidence` values in the code
- Improve lighting conditions
- Hold fingers more clearly extended or folded
