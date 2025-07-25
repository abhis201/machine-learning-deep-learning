# Machine Learning Course Repository

This repository contains coursework, assignments, projects, and implementations for the Machine Learning course.

## Structure

### 📓 Notebooks

#### Basics
- `numpy_fundamentals_for_data_science.ipynb`: Comprehensive NumPy introduction for data science
- `pytorch_tensor_operations_fundamentals.ipynb`: PyTorch fundamentals and tensor operations
- `neural_network_temperature_conversion.ipynb`: Simple neural network for temperature conversion
- `linear_regression_implementation_analysis.ipynb`: Linear regression implementation and analysis
- `pytorch_data_preprocessing_pipeline.ipynb`: Data loading and preprocessing with PyTorch
- `xgboost_boston_housing_prediction.ipynb`: XGBoost implementation for Boston housing prediction

#### Deep Learning
- `conditional_gan_mnist_digit_generation.ipynb`: Conditional GAN for MNIST digit generation
- `gpt_transformer_fundamentals_oop.ipynb`: GPT and transformer fundamentals with OOP
- `binary_pattern_gan_implementation.ipynb`: Simple binary pattern generation using GANs

#### Computer Vision
- `cifar10_cnn_image_classification.ipynb`: CNN-based image classification on CIFAR-10
- `mnist_convolutional_neural_network.ipynb`: Convolutional Neural Networks for MNIST digits

### 📝 Assignments
- **NumPy Assignment**: Comprehensive NumPy problems and solutions
  - Problem definitions and implementations
  - PDF documentation of solutions

### 🚀 Projects
- **Music Genre Classification**: Complete ML project for music genre prediction
  - Data analysis and preprocessing
  - Feature engineering and model training
  - Visualization and performance analysis
  - Comprehensive project documentation

### 🤖 Models
- `wine_quality_deep_learning.onnx`: Trained deep learning model for wine quality prediction
- `wine_quality_xgboost.onnx`: XGBoost model for wine quality classification

### 📊 Datasets
- `iris_flower_classification_dataset.csv`: Classic Iris dataset for classification experiments
- `ml_training_datasets_archive.zip`: Comprehensive collection of ML training datasets
- `nlp_text_processing_input.txt`: Text data for natural language processing experiments

### 📚 Resources
- `prediction_vs_real_plot.png`: Visualization example for model evaluation
- `weka_installer.dmg`: Weka machine learning toolkit installer

## Technologies and Frameworks

### Programming Languages
- **Python**: Primary language for ML implementations
- **R**: Statistical analysis and data mining

### Libraries and Frameworks
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **Matplotlib/Seaborn**: Data visualization
- **ONNX**: Model deployment and interoperability

### Tools
- **Jupyter Notebooks**: Interactive development environment
- **Weka**: Machine learning workbench
- **Git**: Version control

## Getting Started

### Prerequisites
```bash
# Python 3.8+
# Conda or pip package manager
```

### Installation
```bash
# Create virtual environment
conda create -n ml-course python=3.8
conda activate ml-course

# Install core dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
pip install xgboost onnx onnxruntime jupyter notebook

# For specific projects, additional dependencies may be required
```

### Running the Code
```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to specific notebooks and run cells
# For ONNX models, ensure onnxruntime is installed
```

## Course Topics Covered

### Fundamentals
- Linear and Logistic Regression
- Decision Trees and Random Forests
- Support Vector Machines
- K-Means and Hierarchical Clustering
- Principal Component Analysis (PCA)

### Deep Learning
- Neural Networks and Backpropagation
- Convolutional Neural Networks (CNNs)
- Generative Adversarial Networks (GANs)
- Transfer Learning
- Model Optimization and Regularization

### Specialized Topics
- Computer Vision
- Natural Language Processing basics
- Time Series Analysis
- Ensemble Methods
- Model Evaluation and Validation

### Tools and Deployment
- Model Serialization (ONNX)
- Performance Optimization
- Data Pipeline Design
- Visualization and Interpretation

## Projects Highlights

1. **Music Genre Classification**: End-to-end ML pipeline with feature engineering, model comparison, and deployment
2. **Wine Quality Prediction**: Multiple approaches including deep learning and gradient boosting
3. **Computer Vision**: Image classification with state-of-the-art CNN architectures
4. **Generative Models**: Implementation of GANs for data generation

## Author
Abhishek Singh

## Academic Context
Coursework completed as part of Machine Learning curriculum, covering both theoretical foundations and practical implementations with industry-standard tools and frameworks.
