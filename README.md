# Potato Disease Classification System

This project aims to classify diseases in potato plants using Convolutional Neural Networks (CNN). The system is designed to identify Early Blight and Late Blight, which are significant contributors to economic loss and crop waste in agriculture.

## Introduction

The Potato Disease Classification System utilizes deep learning techniques to help farmers identify and manage potato plant diseases effectively. By leveraging CNNs, the system can analyze images of potato leaves and classify them into categories such as Early Blight, Late Blight, and Healthy.

## Installation

To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/shivankursharma018/Potato-Disease-Classification-System-using-Convolutional-Neural-Networks-CNN.git
cd Potato-Disease-Classification-System-using-Convolutional-Neural-Networks-CNN.git
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Ensure you have the dataset ready. You can download it from [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

2. **Model Training**: Use the provided Jupyter notebook to train the CNN model:

```bash
jupyter notebook Training.ipynb
```

3. **Deployment**: Deploy the trained model using FastAPI to create a backend service for disease classification.

## Dataset

The dataset consists of images of potato leaves categorized into three classes:
- Potato Early Blight
- Potato Late Blight
- Potato Healthy Leaf

## Model Training

The training process involves several steps:
- Importing necessary libraries
- Loading and splitting the dataset
- Data preprocessing
- Building and training the CNN model
- Analyzing the model's performance

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
