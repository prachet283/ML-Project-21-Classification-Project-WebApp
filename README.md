# ML Classification Projects WebApp

This repository contains a web app developed using Streamlit and hosted on Streamlit Cloud. The web app integrates five different classification projects, each utilizing machine learning models to provide accurate predictions. The projects covered are:

- Spam Mail Prediction
- Titanic Survival Prediction
- Wine Quality Prediction
- Loan Status Prediction
- Credit Card Fraud Detection

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset Description](#dataset-description)
5. [Technologies Used](#technologies-used)
6. [Model Development Process](#model-development-process)
7. [Models Used](#models-used)
8. [Model Evaluation](#model-evaluation)
9. [Conclusion](#conclusion)
10. [Deployment](#deployment)
11. [Contributing](#contributing)
12. [Contact](#contact)

## Overview

This web application allows users to select from five different classification projects and get predictions based on the input features. Each project was developed through extensive data analysis and model selection processes, ensuring high accuracy and reliability.

## Installation

To run this project locally, please follow these steps:

1. Clone the repository
2. Navigate to the project directory
3. Install the required dependencies

```bash
git clone <repository_url>
cd <project_directory>
pip install -r requirements.txt
```

## Usage

To start the Streamlit web app, run the following command in your terminal:

```bash
streamlit run streamlit_app.py
```

This will launch the web app in your default web browser. You can then select the desired classification project from the sidebar and input the required features to get a prediction.

## Dataset Description

### Spam Mail Prediction
**Description:** This dataset contains emails labeled as spam or not spam, with features such as email content, length, and specific words used.

### Titanic Survival Prediction
**Description:** This dataset includes information about the passengers on the Titanic, with features such as age, sex, passenger class, and fare, used to predict survival.

### Wine Quality Prediction
**Description:** This dataset contains features like acidity, sugar levels, pH, and alcohol content to predict the quality of wine.

### Loan Status Prediction
**Description:** This dataset includes features such as applicant income, loan amount, credit history, and employment status, used to predict loan approval status.

### Credit Card Fraud Detection
**Description:** This dataset contains transactions made by credit cards, with features such as transaction amount and frequency, used to predict fraudulent transactions.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning Libraries:** Scikit-learn, XGBoost
- **Data Analysis and Visualization:** Pandas, NumPy, Matplotlib, Seaborn

## Model Development Process

Each classification project was developed through the following steps:

1. **Importing the Dependencies**
2. **Exploratory Data Analysis (EDA)**
3. **Data Preprocessing**
    - Handling missing values
    - Handling outliers
    - Label encoding/One-hot encoding
    - Standardizing the data
4. **Model Selection**
    - Selected the most common 5 classification models
    - Trained each model and checked cross-validation scores
    - Chose the top 3 models based on cross-validation scores
5. **Model Building and Evaluation**
    - Selected best features using Recursive Feature Elimination (RFE)
    - Performed hyperparameter tuning using Grid Search CV
    - Built the final model with the best hyperparameters and features
    - Evaluated the model using classification reports

## Models Used

The top 3 models for each classification project are as follows:

### Spam Mail Prediction
- Support Vector Classifier: Effective in high-dimensional spaces.
- XGBoost: Boosting algorithm known for high performance.
- Random Forest Classifier: Ensemble method that reduces overfitting.

### Titanic Survival Prediction
- Logistic Regression: Interpretable and performs well with classification.
- XGBoost: Boosting algorithm known for high performance.
- K-Nearest Neighbour: Simple algorithm that works well with small datasets.

### Wine Quality Prediction
- Logistic Regression: Interpretable and performs well with classification.
- XGBoost: Boosting algorithm known for high performance.
- K-Nearest Neighbour: Simple algorithm that works well with small datasets.

### Loan Status Prediction 
- XGBoost: Excellent performance with complex datasets.
- Random Forest Classifier: Robust and handles missing values well.
- Logistic Regression: Highly interpretable and performs well with binary classification.

### Credit Card Fraud Detection
- XGBoost: Powerful gradient boosting framework.
- Random Forest Classifier: Ensemble method that reduces overfitting.
- Support Vector Classifier: Effective in high-dimensional spaces.

## Model Evaluation

### Spam Mail Prediction Model Accuracy
- Support Vector Classifier: 98.21%
- XGBoost: 98.21%
- Random Forest Classifier: 96.59%

### Titanic Survival Prediction Model Accuracy 
- Logistic Regression: 81.00%
- XGBoost: 79.33%
- K-Nearest Neighbour: 78.21%

### Wine Quality Prediction Model Accuracy
- Logistic Regression: 67.50%
- XGBoost: 66.25%
- K-Nearest Neighbour: 58.44%

### Loan Status Prediction Model Accuracy 
- XGBoost: 99.30%
- Random Forest Classifier: 98.83%
- Logistic Regression: 95.55%

### Credit Card Fraud Detection Model Accuracy 
- XGBoost: 92.38%
- Random Forest Classifier: 91.88%
- Support Vector Classifier: 91.37%

## Conclusion

This ML Classification Projects WebApp provides an easy-to-use interface for predicting various outcomes based on input features. The models used are well-validated and tuned for high accuracy. The system aims to assist in decision-making and classification tasks across different domains.

## Deployment

The web app is hosted on Streamlit Cloud. You can access it using the following link:

[ML Classification Projects WebApp](https://ml-project-21-classification-project-webapp-hfcyjreiafqnkaqh3m.streamlit.app/)

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## Contact

If you have any questions or suggestions, feel free to contact me at prachetpandav283@gmail.com.
