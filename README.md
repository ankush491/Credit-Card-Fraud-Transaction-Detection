# Credit-Card-Fraud-Transaction-Detection

#Credit Card Fraud Transaction Detection:

This project focuses on detecting fraudulent credit card transactions using various machine learning classifiers. The goal is to identify suspicious transactions based on historical data, helping banks and financial institutions to prevent and reduce fraud. The project employs different classification algorithms to predict whether a given transaction is fraudulent or legitimate.

#Classifiers Used:

1. Logistic Regression: Logistic Regression is a simple yet powerful linear model used for binary classification problems. In this project, it is applied to predict whether a transaction is fraudulent or not based on the provided features. It calculates the probabilities using the logistic function, and the class with the higher probability is chosen as the predicted label.

2. K-Nearest Neighbors (KNN): KNN is a non-parametric algorithm used for classification and regression tasks. It works by identifying the K-nearest data points to a test instance and predicting the label based on the majority class. In this project, KNN is used to classify transactions by examining their proximity to others in the feature space, offering a simple but effective approach to fraud detection.

3. Support Vector Classifier (SVC): SVC is a powerful classification algorithm that aims to find the hyperplane that best separates different classes in the feature space. By mapping the input data to higher dimensions using a kernel trick, SVC is capable of handling non-linear relationships between features, making it highly effective for fraud detection tasks, where the decision boundaries may not be linear.

4. Decision Tree Classifier: A Decision Tree is a flowchart-like tree structure where each node represents a feature, each branch represents a decision rule, and each leaf node represents the outcome. This classifier works by recursively splitting the data based on feature values to predict the likelihood of fraud in a transaction. It is particularly useful for understanding which features are most important for classification and provides interpretability for model decision-making.

#Techniques Used:

!. ROC Curve: The Receiver Operating Characteristic (ROC) curve is used to evaluate the performance of classification models. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) for various thresholds, helping us assess how well the model distinguishes between fraudulent and legitimate transactions. The area under the ROC curve (AUC) is also computed, with a higher AUC indicating a better model.

2. Principal Component Analysis (PCA) Scatterplot: PCA is a dimensionality reduction technique that transforms features into a lower-dimensional space while retaining the most important information. It helps in visualizing high-dimensional data in 2D or 3D. A PCA scatterplot is used to visualize the distribution of the transactions (fraudulent vs. non-fraudulent) across the first two principal components, providing insights into how well the classifiers separate the classes.

3. GridSearchCV: GridSearchCV is used to tune the hyperparameters of the classifiers to optimize their performance. It performs an exhaustive search over a specified parameter grid and selects the best combination of hyperparameters based on cross-validation results. This ensures that the models are tuned to provide the best possible performance for detecting fraudulent transactions.

#Workflow Overview:

Data Preprocessing: The dataset is cleaned and preprocessed by handling missing values, normalizing the data, and encoding categorical features. Additionally, techniques like oversampling or undersampling may be applied to handle class imbalance, as fraudulent transactions are often much less frequent than legitimate ones.

Model Training and Evaluation: Different classifiers (Logistic Regression, KNN, SVC, Decision Tree) are trained on the preprocessed data. The models are evaluated using metrics like accuracy, precision, recall, F1 score, and ROC-AUC. The effectiveness of each classifier is visualized using ROC curves and compared.

Model Tuning: GridSearchCV is applied to fine-tune the hyperparameters of each classifier. This ensures that each model is optimized for the best performance, particularly in identifying fraudulent transactions.

Dimensionality Reduction and Visualization: PCA is applied to reduce the dimensionality of the feature space, making it easier to visualize the data and assess the decision boundaries of the classifiers using scatterplots. This also helps in understanding how well the classifiers can separate fraudulent and non-fraudulent transactions in lower dimensions.

Key Insights:
Model Comparison: By evaluating the ROC curves and comparing AUC values, we can determine which classifier performs the best in identifying fraudulent transactions. This allows for a more informed choice of algorithm in practical applications.

PCA Scatterplot: PCA helps in understanding how the models perform in different dimensions and how well they can separate the classes.

Hyperparameter Tuning: GridSearchCV ensures that each model is tuned for optimal performance, leading to more accurate predictions of fraudulent transactions.

#Conclusion:

This project demonstrates the power of multiple machine learning classifiers in detecting credit card fraud. By combining various classification algorithms, hyperparameter tuning, and visualization techniques, we can build an effective fraud detection system that helps to reduce financial risks and prevent fraudulent activities.
