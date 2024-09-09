# Project Overview

This project utilizes a machine learning pipeline to predict the "is_hazardous" label of NEOs based on their features, such as absolute magnitude, diameter, relative velocity, and miss distance. Accurate identification of hazardous NEOs is vital for planetary defense and can aid NASA in future observational efforts.
Objectives:

Prepare the dataset for modeling by applying preprocessing techniques.
Address the issue of class imbalance using SMOTE.
Train multiple machine learning models and evaluate their performance.
Use evaluation metrics like Precision, Recall, F1-Score, and ROC-AUC to select the best model.

# Dataset Information

The dataset used for this project includes 338,199 observations of NEOs from NASA. Here are the key columns:

neo_id: Unique identifier for the NEO.
name: Name of the NEO.
absolute_magnitude: Brightness of the NEO.
estimated_diameter_min/max: Minimum and maximum estimated diameter of the NEO.
orbiting_body: The celestial body the NEO is orbiting (e.g., Earth).
relative_velocity: The velocity of the NEO relative to Earth.
miss_distance: How close the NEO came to Earth.
is_hazardous: Target variable indicating if the NEO is classified as hazardous (True/False).

# Data Preprocessing
Key Preprocessing Steps:

Handling Missing Values: The dataset did not contain missing values.
Feature Selection: Selected key features based on domain knowledge and correlation analysis.
Categorical Encoding: Encoded the orbiting_body column using one-hot encoding.
Feature Scaling: Scaled the numerical features (e.g., relative_velocity, miss_distance) using StandardScaler.
Addressing Class Imbalance: Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the hazardous (minority) and non-hazardous (majority) classes.

# Modeling

The following machine learning models were trained on the preprocessed data:

Random Forest Classifier (chosen as the final model due to its performance)

Hyperparameters:

Random Forest:
        n_estimators: 100

        class_weight: 'balanced'

        random_state: 42

# Evaluation Metrics

The models were evaluated using the following metrics:

Precision: Measures the accuracy of the positive predictions.
Recall: Measures how many actual positives were captured.
F1-Score: The harmonic mean of precision and recall.
ROC-AUC: Area Under the Receiver Operating Characteristic curve.

Random Forest Performance:

Precision: 0.92
Recall: 0.85
F1-Score: 0.88
ROC-AUC: 0.94

# Findings and Insights

Feature Importance: The relative_velocity, miss_distance, and absolute_magnitude were the most influential features for predicting hazardous NEOs. 
Class Imbalance: The original dataset had a significant imbalance between hazardous and non-hazardous NEOs. The application of SMOTE improved model performance on the minority class.
Random Forest Model: Provided the best balance between precision and recall, making it ideal for this use case where false negatives (i.e., missing hazardous NEOs) are critical.

 # Installation & Usage
Requirements:

   Python 3.x ,
   Libraries:
        pandas,
        ,numpy
        ,scikit-learn
        ,matplotlib
        ,seaborn
        ,imblearn
