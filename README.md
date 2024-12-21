# Paint-A-Distance

**Paint-A-Distance** is an interactive drawing application that allows you to draw in real-time using hand gestures. The application uses computer vision and hand tracking to detect gestures and perform actions like drawing, pausing, and clearing the canvas.

## Features

- **1 Finger:** Draw on the screen.
- **2 Fingers:** Pause the drawing.
- **5 Fingers:** Clear the screen.
- **Color Options:** Choose between three different drawing colors.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paint-a-distance.git
   cd paint-a-distance
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.7+
- OpenCV
- Skimage
- NumPy

## Usage

1. Run the application:
   ```bash
   python3 main_script.py
   ```

2. Use hand gestures to draw and interact:
   - **1 Finger:** Start drawing.
   - **2 Fingers:** Pause the drawing.
   - **5 Fingers:** Clear the canvas.
   - **Color Options:** Switch colors from the provided menu.

## Directory Structure

```
paint-a-distance/
├── app.py              # Main application script
├── README.md           # Project documentation
├── requirements.txt    # Required dependencies
├── assets/             # Images and assets
├── utils/              # Utility functions
└── models/             # Pre-trained models (if any)
```
