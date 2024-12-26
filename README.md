# Face Recognition Using Eigenfaces

This repository implements face recognition using the Eigenface technique. Eigenfaces leverage Principal Component Analysis (PCA) to represent human faces as a set of basis images, reducing dimensionality while preserving key features for recognition. The approach is effective for identifying individuals from images with high accuracy.

- An eigenface is derived from the eigenvectors of the covariance matrix of a dataset of face images. These eigenfaces form a basis for representing faces in a reduced dimensional space. Sirovich and Kirby popularized this approach, and later advanced by Matthew Turk and Alex Pentland. The key benefits of using eigenfaces include dimensionality reduction and effective classification of faces.

## Steps in Face Recognition Using Eigenfaces

- The goal is to identify individuals from images by comparing the features (eigenfaces) of input faces with those of known individuals.

###  Workflow

#### a. Import Necessary Libraries
Libraries such as `numpy`, `matplotlib`, and `scikit-learn` are required for this project . 

#### b. Load the Dataset
The Olivetti dataset, a standard for face recognition tasks, is used as a benchmark dataset.
The dataset can be found on Kaggle :https://www.kaggle.com/datasets/shuchirb/olivetti-faces

#### c. Split the Dataset
Split the data into training and test sets to evaluate model performance.

#### d. Preprocess the Data
Normalize or standardize pixel values to improve PCA's performance.

#### e. Apply PCA
- Use PCA to reduce the dimensionality of the dataset.
- Select the number of components (`n_components`) to retain sufficient variance.
- Transform both training and test data consistently using the learned principal components.

#### f. Train a Classifier
- Use an SVM (Support Vector Machine) classifier to map PCA-transformed features to class labels.


## Confusion Matrix
A confusion matrix evaluates the classification model's performance by comparing predicted and actual values:

##Explain the classification report: The model achieved an accuracy of 0.99, meaning 99% of the total predictions were correct.
Metrics
•	Precision: The ratio of correctly predicted positive observations to the total predicted positives. A precision of 1.00 means all predictions for that class were correct.
•	Recall (Sensitivity): The ratio of correctly predicted positive observations to all observations in the actual class. It reflects the model's ability to identify all actual positives. A recall of 1.00 means the model identified all instances of that class correctly.
•	F1-Score: The harmonic mean of precision and recall, providing a balance between the two. An F1-score of 1.00 indicates perfect precision and recall.
•	Support: The number of actual occurrences of each class in the dataset. It shows how many instances there are of each class.


## Key Features of the Implementation
- Dimensionality Reduction: PCA ensures efficient data representation by reducing redundant information.
- High Accuracy: The model achieves up to 99% accuracy on the Olivetti dataset.
- Visualization Tools: Insights into eigenfaces and classification outcomes.

## References
- [Scikit-learn Documentation on PCA and SVM](https://scikit-learn.org/0.15/auto_examples/applications/face_recognition.html)
- Sirovich, L., & Kirby, M. (1987). Low-dimensional procedure for the characterization of human faces.
- Turk, M., & Pentland, A. (1991). Eigenfaces for recognition.


Feel free to check the full notebook on Kaggle : https://www.kaggle.com/code/nawreshamrouni/face-recognition-using-eigenfaces
