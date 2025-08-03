# 🗂️ Waste Classification System

A deep learning-powered waste classification system built with TensorFlow and Streamlit for automated waste sorting and environmental sustainability.

## 🎯 Overview

This project implements an intelligent waste classification system that can automatically categorize waste items into 6 different categories: **Cardboard**, **Glass**, **Metal**, **Paper**, **Plastic**, and **Trash**. The system uses transfer learning with ResNet50 architecture and provides a user-friendly web interface for real-time classification.

## ✨ Key Features

- **🤖 Advanced AI Model**: ResNet50-based transfer learning with 47.5% validation accuracy
- **🎨 Interactive Web App**: Professional Streamlit interface with drag-and-drop image upload
- **📊 Real-time Results**: Instant classification with confidence scores and recycling tips
- **🔍 Comprehensive Evaluation**: Detailed model analysis with confusion matrices and metrics
- **📁 Professional Structure**: Clean, organized codebase ready for production deployment

## 🛠️ Technology Stack

### **Core Technologies**
| Technology | Purpose | Version |
|------------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | **Primary Language** | 3.8+ |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | **Deep Learning Framework** | 2.20.0+ |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | **High-level Neural Networks API** | Integrated with TF |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | **Web Application Framework** | 1.47.0+ |

### **Machine Learning & AI**
- **🧠 Transfer Learning**: ResNet50 pre-trained on ImageNet
- **🎯 Computer Vision**: Image classification and preprocessing
- **📊 Model Evaluation**: Confusion matrices, classification reports
- **🔧 Data Augmentation**: Rotation, flipping, zooming techniques
- **⚡ Optimization**: Adam optimizer with learning rate scheduling

### **Data Science & Analytics**
| Library | Purpose | Key Features |
|---------|---------|--------------|
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | **Numerical Computing** | Array operations, mathematical functions |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | **Data Manipulation** | Data analysis and preprocessing |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white) | **Static Visualization** | Plots, charts, confusion matrices |
| ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white) | **Statistical Visualization** | Advanced statistical plots |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) | **Interactive Visualization** | Dynamic charts and graphs |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | **Machine Learning Utilities** | Model evaluation, preprocessing |

### **Computer Vision & Image Processing**
- **📷 OpenCV**: Image preprocessing and computer vision operations
- **🖼️ Pillow (PIL)**: Image manipulation and format handling
- **🎨 Image Augmentation**: Real-time data augmentation pipeline
- **📐 Preprocessing**: Resizing, normalization, color space conversion

### **Development & DevOps**
| Tool | Purpose | Usage |
|------|---------|-------|
| ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) | **Version Control** | Source code management |
| ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) | **Repository Hosting** | Code collaboration and deployment |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | **Interactive Development** | Data exploration and prototyping |
| ![PyTest](https://img.shields.io/badge/PyTest-0A9EDC?style=flat&logo=pytest&logoColor=white) | **Testing Framework** | Unit testing and validation |

### **Architecture & Design Patterns**
- **🏗️ Modular Architecture**: Separation of concerns with organized modules
- **🔧 Object-Oriented Programming**: Class-based model and utility design
- **📦 Package Management**: Proper Python packaging with `__init__.py`
- **🎯 MVC Pattern**: Model-View-Controller architecture implementation
- **⚙️ Configuration Management**: Centralized config handling

### **Performance & Optimization**
- **🚀 Model Optimization**: Efficient inference with pre-trained weights
- **💾 Memory Management**: Optimized image loading and processing
- **⚡ Caching**: Streamlit caching for improved performance
- **🎛️ Hyperparameter Tuning**: Systematic optimization approach

## 🚀 Performance Metrics

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 47.5% |
| **Training Images** | 4,041 |
| **Test Images** | 1,013 |
| **Classes** | 6 waste categories |
| **Architecture** | ResNet50 + Custom head |

## 📋 Requirements

### System Requirements
- Python 3.8+
- Windows/macOS/Linux
- 4GB+ RAM recommended
- GPU support optional (CUDA-compatible)

### Dependencies
```
tensorflow>=2.20.0
streamlit>=1.47.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
pillow>=8.3.0
scikit-learn>=1.0.0
plotly>=5.0.0
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd WasteManagement
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model (if not included)
The trained model `improved_waste_classifier.h5` should be in the root directory.

## 🎮 Usage

### Running the Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`

### Model Evaluation
```bash
python scripts/evaluate_improved_model.py
```

### Running Tests
```bash
python tests/test_improved_model.py
```

### Project Structure Validation
```bash
python scripts/validate_structure.py
```

## 📁 Project Structure

```
WasteManagement/
├── 📱 app.py                          # Main Streamlit application
├── 📋 requirements.txt                # Project dependencies
├── 📖 README.md                       # Project documentation
├── 🤖 improved_waste_classifier.h5    # Trained model file
├── 📂 src/                            # Core source code
│   ├── config.py                      # Configuration settings
│   ├── model.py                       # Model architecture & prediction
│   ├── data_processor.py              # Data processing utilities
│   ├── utils.py                       # Helper functions
│   ├── visualizer.py                  # Visualization tools
│   └── __init__.py                    # Package initialization
├── 📊 results/                        # Model outputs & visualizations
│   ├── confusion_matrix.png           # Performance metrics
│   └── training_history.png           # Training progress charts
├── 💾 data/                           # TrashNet dataset
│   └── data/                          # Organized train/test splits
│       ├── train/                     # Training images by category
│       └── test/                      # Testing images by category
```

## 🎯 Waste Categories

The system classifies waste into 6 categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **📦 Cardboard** | Corrugated and flat cardboard | Boxes, packaging materials |
| **🥃 Glass** | Glass containers and bottles | Jars, bottles, glassware |
| **🔩 Metal** | Metal cans and containers | Aluminum cans, steel containers |
| **📄 Paper** | Paper products and documents | Newspapers, magazines, documents |
| **🥤 Plastic** | Plastic containers and packaging | Bottles, containers, plastic bags |
| **🗑️ Trash** | Non-recyclable waste items | Mixed waste, contaminated items |

## 💡 Model Architecture

### Transfer Learning Approach
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Custom Head**: Dense layers with BatchNormalization and Dropout
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: L2 regularization and data augmentation

### Training Configuration
- **Epochs**: 40
- **Batch Size**: 32
- **Image Size**: 224x224 pixels
- **Data Augmentation**: Rotation, flip, zoom, shift

## 📊 Model Performance

### Classification Report
```
              precision    recall  f1-score   support
   cardboard       0.45      0.52      0.48       119
       glass       0.52      0.48      0.50       164
       metal       0.46      0.45      0.45       140
       paper       0.48      0.51      0.49       185
     plastic       0.49      0.47      0.48       192
       trash       0.45      0.43      0.44       213

    accuracy                           0.47      1013
   macro avg       0.48      0.48      0.47      1013
weighted avg       0.48      0.47      0.47      1013
```

## 🚀 Deployment

### Local Deployment
The application is ready to run locally using the Streamlit command above.

### Production Deployment
For production deployment, consider:
- **Docker**: Containerize the application
- **Cloud Platforms**: Deploy to AWS, GCP, or Azure
- **Streamlit Cloud**: Direct deployment to Streamlit's cloud platform

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **TrashNet Dataset**: Original dataset for waste classification

**🌍 Making waste management smarter, one classification at a time! ♻️**
