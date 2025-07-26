# Bangla Sign Language Recognition using Hand Gestures

## Overview
This project provides a complete pipeline for recognizing Bangla Sign Language (BdSL) hand gestures using deep learning and computer vision. It includes real-time detection, model training, data preprocessing, and evaluation tools, enabling both research and practical applications for Bangla sign language recognition.

## Features
- **Real-time Detection:** Recognize BdSL hand gestures from webcam video using a trained CNN model.
- **Multiple Model Architectures:** Includes CNN, MobileNetV2, and VGG16 models for experimentation and benchmarking.
- **Data Preprocessing & Augmentation:** Scripts and notebooks for preparing and augmenting custom datasets.
- **Jupyter Notebooks:** For model training, evaluation, and testing.
- **Bangla Font Support:** Uses a custom Bangla font for accurate label rendering in detection results.

## Directory Structure
```
.
├── Real Time Detection/
│   ├── bdsl.py              # Real-time detection script
│   └── kalpurush.ttf        # Bangla font for label rendering
├── cnn_model.keras          # Trained CNN model
├── mobilenet_model.keras    # Trained MobileNetV2 model
├── vgg16_model.keras        # Trained VGG16 model
├── dataset-1-train-thesis.ipynb      # Training on BdSL47 dataset
├── dataset-2-train-thesis.ipynb      # Training on 30/40 words dataset
├── final-dataset-train-thesis.ipynb  # Training on final 77-class dataset
├── final-dataset-testing.ipynb       # Model testing and visualization
├── imageprocessing.ipynb             # Data augmentation and preprocessing
├── links_info.txt           # Dataset and demo video links
└── README.md                # Project documentation
```

## Datasets
- **BdSL47 Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/sada335/bdsl-47-dataset)
- **30 Words Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/abirmunna/bangla-sign-language-40)
- **Final 77-Class Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/sabbirmusfique63/dataset-of-77-bangla-sign)
- **Demo Video:** [Google Drive Link](https://drive.google.com/file/d/1WnnyD8PtuxmWtMid_RwFYwE8tEXCQzbn/view?usp=sharing)

## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/sabbir-063/BdSL-Codes.git
cd BdSL-Codes
```

### 2. Install Dependencies
Install required Python packages:
```bash
pip install tensorflow opencv-python mediapipe pillow numpy scikit-learn matplotlib seaborn lightgbm patool dask
```

### 3. Download Datasets & Models
- Download datasets from the provided Kaggle links and place them as described in the notebooks.
- Ensure pretrained models (`cnn_model.keras`, etc.) are present in the project root.
- Place `kalpurush.ttf` in the `Real Time Detection` directory.

### 4. Train Models (Optional)
Use the provided Jupyter notebooks to train or fine-tune models on your dataset.

### 5. Real-Time Detection
Run the real-time detection script:
```bash
cd "Real Time Detection"
python bdsl.py
```
- Ensure your webcam is connected.
- Press `q` to quit the detection window.

## Notebooks
- **Training:**
  - `final-dataset-train-thesis.ipynb`: Training on the final 77-class dataset.
  - `dataset-1-train-thesis.ipynb`: Training on BdSL47 dataset.
  - `dataset-2-train-thesis.ipynb`: Training on 30/40 words dataset.
- **Testing:**
  - `final-dataset-testing.ipynb`: Test and visualize predictions on sample images.
- **Preprocessing:**
  - `imageprocessing.ipynb`: Data augmentation and preprocessing scripts.

## Model Files
- `cnn_model.keras`, `mobilenet_model.keras`, `vgg16_model.keras`: Pretrained models for detection and evaluation.

## Dependencies
- Python 3.7+
- TensorFlow
- OpenCV
- MediaPipe
- Pillow
- NumPy
- scikit-learn
- matplotlib
- seaborn
- lightgbm
- patool
- dask

## Credits
- **Datasets:**
  - [BdSL47 Dataset](https://www.kaggle.com/datasets/sada335/bdsl-47-dataset)
  - [30 Words Dataset](https://www.kaggle.com/datasets/abirmunna/bangla-sign-language-40)
- **Demo Video:** [Google Drive](https://drive.google.com/file/d/1WnnyD8PtuxmWtMid_RwFYwE8tEXCQzbn/view?usp=sharing)
- **Font:** [kalpurush.ttf](https://www.omicronlab.com/fonts/kalpurush.html)

## License
This project is for academic and research purposes. Please cite the dataset sources and respect their licenses.

---
For questions or contributions, please open an issue or submit a pull request.
