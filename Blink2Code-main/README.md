# 👁️ Blink2Code

**Blink2Code** is an innovative assistive communication system that converts eye blinks into Morse code and translates them into readable text. This project leverages computer vision, deep learning, and real-time video processing to enable hands-free communication through eye movements.

---

## 🌟 Features

- **Real-time Blink Detection**: Uses webcam to detect and classify short and long blinks
- **Morse Code Translation**: Converts blink patterns into Morse code sequences
- **Text Output**: Translates Morse code into readable text
- **CNN-based Classification**: Deep learning model for accurate blink detection
- **Eye Tracking**: Uses facial landmark detection for precise eye monitoring
- **File Output**: Saves translated text to `translated_text.txt` for later reference

---

## 🎯 Use Cases

- **Assistive Technology**: Communication aid for individuals with limited mobility
- **Accessibility**: Alternative input method for hands-free interaction
- **Research**: Study of eye-based human-computer interaction
- **Educational**: Demonstrates computer vision and deep learning applications

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.12**
- **OpenCV**: Real-time video processing and face detection
- **dlib**: Facial landmark detection (68-point facial landmarks)
- **PyTorch**: Deep learning framework for CNN model
- **NumPy**: Numerical computations

### Key Components
1. **Face Detection**: dlib's frontal face detector
2. **Landmark Detection**: 68-point facial landmark predictor
3. **EAR Calculation**: Eye Aspect Ratio for blink detection
4. **CNN Model**: Custom neural network for blink classification
5. **Morse Translator**: Pattern-to-text conversion system

---

## 📂 Project Structure

```
Blink2Code/
├── blink_detection.py          # Main application - real-time blink detection
├── cnn_model.py                 # CNN architecture for blink classification
├── morse_translator.py          # Morse code translation module
├── train.py                     # Model training script
├── preprocess.py                # Data preprocessing utilities
├── evaluate.py                  # Model evaluation script
├── models/                      # Trained model directory
│   └── blink_cnn.pth           # Trained CNN weights
├── dataset/                     # Training dataset
├── shape_predictor_68_face_landmarks.dat  # dlib facial landmark model
├── deploy.prototxt              # DNN face detector configuration
├── res10_300x300_ssd_iter_140000.caffemodel  # DNN face detector weights
└── translated_text.txt          # Output file with translations
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- Webcam
- Windows/Linux/MacOS

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Blink2Code.git
cd Blink2Code
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install dlib  # Or install from wheel file
```

**Note**: If you encounter issues installing dlib, use the provided wheel file:
```bash
pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
```

### Step 3: Download Required Models

Download the dlib facial landmark predictor:
```bash
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract and place in the project root directory
```

---

## 💻 Usage

### Running the Application

1. **Start the blink detection system**:
```bash
python blink_detection.py
```

2. **Position yourself in front of the webcam**
   - Ensure good lighting
   - Face the camera directly
   - Keep your face clearly visible

3. **Blink to communicate**:
   - **Short Blink** (< 0.3 seconds) = `.` (dot in Morse code)
   - **Long Blink** (> 0.3 seconds) = `-` (dash in Morse code)
   - **Pause** (1.5 seconds) = End of character/word

4. **View translations**:
   - Real-time output appears in the console
   - All translations are saved to `translated_text.txt`

5. **Exit**: Press `q` to quit the application

---

## 📊 How It Works

### 1. **Face & Eye Detection**
- dlib detects faces in each video frame
- Extracts 68 facial landmarks including eye coordinates
- Tracks left eye (points 36-41) and right eye (points 42-47)

### 2. **Eye Aspect Ratio (EAR) Calculation**
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
- Measures eye openness
- Drops below threshold (0.25) when eye is closed
- Used to detect blink events

### 3. **Blink Classification**
- **Short Blink**: Duration < 0.3 seconds → Morse dot (.)
- **Long Blink**: Duration > 0.3 seconds → Morse dash (-)
- CNN model enhances classification accuracy

### 4. **Morse Code Translation**
- Blinks are accumulated into sequences
- After 1.5-second pause, sequence is processed
- Morse code is converted to text using dictionary lookup

### 5. **Output**
- Displays on-screen feedback
- Prints to console
- Appends to `translated_text.txt`

---

## 🧠 CNN Model Architecture

```python
BlinkCNN(
  (fc1): Linear(8 → 128)
  (relu): ReLU()
  (fc2): Linear(128 → 64)
  (fc3): Linear(64 → 2)
  (softmax): Softmax(dim=1)
)
```

- **Input**: 8 features (eye landmarks)
- **Hidden Layers**: 128 → 64 neurons
- **Output**: 2 classes (blink/no-blink)
- **Training**: 20 epochs, Adam optimizer, CrossEntropy loss

---

## 📝 Morse Code Reference

### Letters
```
A: .-    B: -...  C: -.-.  D: -..   E: .
F: ..-.  G: --.   H: ....  I: ..    J: .---
K: -.-   L: .-..  M: --    N: -.    O: ---
P: .--.  Q: --.-  R: .-.   S: ...   T: -
U: ..-   V: ...-  W: .--   X: -..-  Y: -.--
Z: --..
```

### Numbers
```
1: .----  2: ..---  3: ...--  4: ....-  5: .....
6: -....  7: --...  8: ---..  9: ----.  0: -----
```

### Example Usage
- **"SOS"**: Short-Short-Short, Long-Long-Long, Short-Short-Short
- **"HI"**: Short-Short-Short-Short (pause) Short-Short

---

## ⚙️ Configuration

### Adjustable Parameters in `blink_detection.py`:

```python
BLINK_THRESHOLD = 0.25           # EAR threshold for blink detection
BLINK_CONSEC_FRAMES = 1          # Consecutive frames to confirm blink
SHORT_BLINK_DURATION = 0.3       # Max duration for short blink (seconds)
MORSE_GAP_DURATION = 1.5         # Pause duration to end character (seconds)
```

---

## 🔧 Training the Model

To train the CNN model on your own dataset:

```bash
python train.py
```

- Loads preprocessed data from `dataset/`
- Trains for 20 epochs
- Saves model to `models/blink_cnn.pth`
- Displays training loss every 5 epochs

---

## 🎥 Demo

**Example Workflow**:
1. Launch application
2. Blink: Short-Short-Short (S)
3. Wait 1.5 seconds
4. Console output: `Morse: ... → Text: S`
5. Text saved to `translated_text.txt`

---

## 🐛 Troubleshooting

### Camera Not Detected
```
Error: Could not access webcam
```
**Solution**: Check camera permissions and ensure no other application is using it

### Model File Not Found
```
Error: Model file 'shape_predictor_68_face_landmarks.dat' not found
```
**Solution**: Download from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

### No Face Detected
```
No face detected, skipping frame
```
**Solution**: Improve lighting, move closer to camera, ensure face is visible

### Blinks Not Registering
- Adjust `BLINK_THRESHOLD` (try 0.20 or 0.30)
- Reduce `SHORT_BLINK_DURATION` for faster blinks
- Check EAR values in video feed

---

## 📈 Future Enhancements

- [ ] Add support for multiple users
- [ ] Implement word prediction/autocomplete
- [ ] Create GUI for easier interaction
- [ ] Add calibration mode for personalized thresholds
- [ ] Support for different languages
- [ ] Mobile application version
- [ ] Cloud-based text storage

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **dlib**: Face detection and facial landmark prediction
- **OpenCV**: Computer vision and video processing
- **PyTorch**: Deep learning framework
- **Morse Code Community**: For the universal communication standard

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please reach out:

- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
- **Email**: your.email@example.com

---

## ⭐ Star This Repo!

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for accessible communication**
