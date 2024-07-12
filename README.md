# Malaria-diagnosis

### Overview
This project aims to develop a deep learning model to diagnose malaria using microscopic images of blood smears. The model is built using TensorFlow and trained on the Malaria dataset from TensorFlow Datasets. The primary goal is to accurately classify images as either 'infected' or 'uninfected'. <br>
### Dataset
The dataset used is the Malaria dataset from TensorFlow Datasets, which contains 27,558 images of blood smears with labels indicating whether the sample is infected with malaria or not. <br>
### Steps Involved <br>
1. Data Loading
The dataset is loaded using TensorFlow Datasets, which provides a convenient way to access and preprocess the data.

2. Data Splitting
The dataset is split into training, validation, and test sets to train the model and evaluate its performance.

3. Data Visualization
Sample images from the dataset are visualized to understand the data distribution and characteristics. The labels are displayed to indicate whether the samples are infected or uninfected.

4. Data Processing
The images are resized and rescaled to a standard size and range. This step includes normalization and standardization to prepare the data for training.

5. Model Building
A convolutional neural network (CNN) is constructed using TensorFlow. The model architecture includes several layers such as convolutional layers, max-pooling layers, batch normalization, and dense layers. The final layer uses a sigmoid activation function to output a binary classification.

6. Model Training
The model is compiled with an optimizer, loss function, and evaluation metrics. It is then trained on the training dataset, with validation on the validation set to monitor performance and adjust parameters.

7. Model Evaluation
The trained model is evaluated on the test dataset to assess its accuracy and performance in diagnosing malaria.

8. Model Saving and Loading
The trained model and its weights are saved in TensorFlow's native format for future use. The model can be loaded back for further predictions or evaluations.

9. Prediction and Visualization
The model's predictions are visualized on sample test images to compare the predicted labels with the actual labels. This helps in understanding the model's performance qualitatively.
