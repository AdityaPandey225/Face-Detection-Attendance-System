# Face Detection Attendance System

A real-time face detection and recognition system built with OpenCV that can identify and label faces from trained images. This system uses machine learning to recognize faces and display person names with confidence scores.

## 🎯 Features

- **Real-time Face Detection**: Detects faces from webcam feed using Haar Cascade Classifier
- **Face Recognition**: Recognizes trained faces and displays person names with confidence percentage
- **Flexible Training**: Supports two methods for organizing training images:
  - **Folder-based**: Organize images in separate folders (one folder per person)
  - **File-based**: Name files as `PersonName_number.jpg` (e.g., `utkarsh_1.jpg`)
- **Sample Collection**: Option to collect face samples directly from camera
- **Label Display**: Shows recognized person's name and confidence score in real-time
- **Multiple Exit Options**: Press 'q', ESC, or Enter to exit the detection window

## 📋 Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- OpenCV Contrib (`opencv-contrib-python`) - for face recognition module
- NumPy

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/AdityaPandey225/Face-Detection-Attendance-System.git
cd Face-Detection-Attendance-System
```

2. Install required packages:
```bash
pip install opencv-python opencv-contrib-python numpy
```

## 📖 Usage

### Method 1: Train from Uploaded Photos (Recommended)

1. **Organize your training images** using one of these methods:

   **Option A - Folder-based:**
   ```
   faces/
     ├── John/
     │   ├── photo1.jpg
     │   ├── photo2.jpg
     │   └── photo3.jpg
     ├── Mary/
     │   ├── photo1.jpg
     │   └── photo2.jpg
   ```

   **Option B - File-based:**
   ```
   faces/
     ├── John_1.jpg
     ├── John_2.jpg
     ├── Mary_1.jpg
     └── Mary_2.jpg
   ```

2. **Run the program:**
```bash
python faceDetection.py
```

3. The program will:
   - Train the model from images in the `faces/` folder
   - Open your webcam for real-time face recognition
   - Display recognized names with confidence scores

### Method 2: Collect Samples from Camera

1. Uncomment the `collect_samples("faces")` line in the main section
2. Run the program to collect face samples directly from your camera
3. Press 'q', ESC, or Enter to stop collection

## 🏗️ Project Structure

```
Face-Detection-Attendance-System/
│
├── faceDetection.py          # Main application file
├── faces/                     # Training images folder
│   ├── utkarsh_1.jpg
│   ├── utkarsh_2.jpg
│   └── ...
├── haarcascade_frontalface_default.xml  # Haar Cascade classifier
└── README.md                  # This file
```

## 🔧 How It Works

1. **Face Detection**: Uses Haar Cascade Classifier to detect faces in real-time video feed
2. **Model Training**: 
   - Loads training images from the `faces/` folder
   - Extracts person names from folder structure or filenames
   - Trains LBPH (Local Binary Patterns Histograms) face recognizer
3. **Face Recognition**:
   - Detects faces in each frame
   - Compares detected face with trained model
   - Displays person name and confidence score if confidence > 82%
   - Shows "Unknown" for unrecognized faces

## 📝 Code Structure

- `collect_samples(data_path)`: Collects face samples from camera
- `train_model(data_path)`: Trains the face recognition model
- `recognize_faces(model, label_to_name)`: Performs real-time face recognition

## 🎮 Controls

- **'q' key**: Exit the application
- **ESC key**: Exit the application
- **Enter key**: Exit the application

## ⚙️ Configuration

- **Confidence Threshold**: Currently set to 82% (can be modified in `recognize_faces()` function)
- **Face Detection Parameters**: `scaleFactor=1.1, minNeighbors=3` (adjustable in code)
- **Image Size**: Training images are resized to 200x200 pixels

## 🔍 Example Output

When a face is recognized, the system displays:
- Green bounding box around the detected face
- Person's name and confidence percentage (e.g., "utkarsh (85%)")
- "Unknown" label for unrecognized faces

## 🛠️ Troubleshooting

- **Camera not opening**: Check if your webcam is connected and not being used by another application
- **No faces detected**: Ensure good lighting and face the camera directly
- **Low recognition accuracy**: Add more training images with different angles and lighting conditions
- **Module not found errors**: Make sure `opencv-contrib-python` is installed (not just `opencv-python`)

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Aditya Pandey**
- GitHub: [@AdityaPandey225](https://github.com/AdityaPandey225)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📚 Technologies Used

- **OpenCV**: Computer vision library for face detection and recognition
- **NumPy**: Numerical computing for image processing
- **Python**: Programming language

---

⭐ If you find this project helpful, please consider giving it a star!

