# Heart Disease Prediction

## Project Overview

The Heart Disease Prediction project is a machine learning model designed to predict the presence of heart disease in patients based on medical attributes. The model is built using logistic regression and trained on a dataset containing various health indicators.

## Dataset

The dataset used in this project is `heart.csv`, which contains 303 records with 14 features:

- `age`: Age of the patient
- `sex`: Gender (1 = Male, 0 = Female)
- `cp`: Chest pain type
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar (1 = True, 0 = False)
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1 = Yes, 0 = No)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels (0-3) colored by fluoroscopy
- `thal`: Thalassemia status
- `target`: Presence of heart disease (1 = Disease, 0 = No Disease)

## Dependencies

To run this project, the following Python libraries are required:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## How to Run the Code

1. Clone the repository or download the project files.
2. Install the required dependencies using:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Run the script:
   ```bash
   python heart_disease_prediction.py
   ```
4. The script will load the dataset, train the model, evaluate accuracy, and allow predictions on new input data.

## Data Preprocessing

- The dataset is loaded using Pandas.
- Checked for missing values (none found).
- Analyzed feature distributions and descriptive statistics.
- The dataset is split into features (`X`) and target (`Y`).
- The data is further divided into training (80%) and testing (20%) sets.

## Model Training

A Logistic Regression model is trained and used.

## Model Evaluation

The model's accuracy is measured on training and testing data:

### Results:

- **Training Accuracy:** \~83%
- **Testing Accuracy:** \~90%

## Predictive System

A simple function is implemented to predict whether a patient has heart disease based on input features.

## Conclusion

This project successfully builds a predictive model for heart disease detection using logistic regression. The model demonstrates high accuracy and provides quick predictions based on patient data.

## Future Improvements

- Experiment with other classification algorithms like Random Forest or Neural Networks.
- Optimize hyperparameters for better accuracy.
- Deploy the model as a web-based application for real-time predictions.
