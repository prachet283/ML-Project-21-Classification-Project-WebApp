# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:25:58 2024

@author: prachet
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import json
import pandas as pd


#loading the saved model of loan status prediction
with open("Preprocessing File/ML-Project-5-Loan_Status_Prediction_Preprocessing_Files/columns.pkl", 'rb') as f:
    all_columns_loan_status = pickle.load(f)
with open("Preprocessing File/ML-Project-5-Loan_Status_Prediction_Preprocessing_Files/cat_columns.pkl", 'rb') as f:
    cat_columns_loan_status = pickle.load(f)
with open("Preprocessing File/ML-Project-5-Loan_Status_Prediction_Preprocessing_Files/encoder.pkl", 'rb') as f:
    encoder_loan_status = pickle.load(f)
with open("Preprocessing File/ML-Project-5-Loan_Status_Prediction_Preprocessing_Files/encoded_columns.pkl", 'rb') as f:
    encoded_columns_loan_status = pickle.load(f)
with open("Preprocessing File/ML-Project-5-Loan_Status_Prediction_Preprocessing_Files/training_columns.pkl", 'rb') as f:
    training_columns_loan_status = pickle.load(f)
with open("Preprocessing File/ML-Project-5-Loan_Status_Prediction_Preprocessing_Files/scaler.pkl", 'rb') as f:
    scaler_loan_status = pickle.load(f)
with open("Best Features/ML-Project-5-Loan_Status_Prediction_Best_Features/best_features_xgb.json", 'r') as file:
    best_features_xgb_loan_status = json.load(file)
with open("Best Features/ML-Project-5-Loan_Status_Prediction_Best_Features/best_features_rfc.json", 'r') as file:
    best_features_rfc_loan_status = json.load(file)
with open("Best Features/ML-Project-5-Loan_Status_Prediction_Best_Features/best_features_svc.json", 'r') as file:
    best_features_svc_loan_status = json.load(file)
with open("Models/ML-Project-5-Loan_Status_Prediction_Models/loan_status_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_loan_status = pickle.load(f)
with open("Models/ML-Project-5-Loan_Status_Prediction_Models/loan_status_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_loan_status = pickle.load(f)
with open("Models/ML-Project-5-Loan_Status_Prediction_Models/loan_status_trained_svc_model.sav", 'rb') as f:
    loaded_model_svc_loan_status = pickle.load(f)


#loading the saved model of wine quality prediction
with open("Preprocessing File/ML-Project-6-Wine_Quality_Prediction_Preprocessing_Files/columns.pkl", 'rb') as f:
    all_features_wine_quality = pickle.load(f)
with open("Preprocessing File/ML-Project-6-Wine_Quality_Prediction_Preprocessing_Files/scaler.pkl", 'rb') as f:
    scalers_wine_quality = pickle.load(f)
with open("Best Features/ML-Project-6-Wine_Quality_Prediction_Best_Features/best_features_rfc.json", 'r') as file:
    best_features_rfc_wine_quality = json.load(file)
with open("Best Features/ML-Project-6-Wine_Quality_Prediction_Best_Features/best_features_xgb.json", 'r') as file:
    best_features_xgb_wine_quality = json.load(file)
with open("Best Features/ML-Project-6-Wine_Quality_Prediction_Best_Features/best_features_knn.json", 'r') as file:
    best_features_knn_wine_quality = json.load(file)
with open("Models/ML-Project-6-Wine_Quality_Prediction_Models/wine_quality_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_wine_quality = pickle.load(f)
with open("Models/ML-Project-6-Wine_Quality_Prediction_Models/wine_quality_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_wine_quality = pickle.load(f)
with open("Models/ML-Project-6-Wine_Quality_Prediction_Models/wine_quality_trained_knn_model.sav", 'rb') as f:
    loaded_model_knn_wine_quality = pickle.load(f)


#loading the saved model of credit card fraud detection
with open("Preprocessing File/ML-Project-10-Credit_Card_Fraud_Detection_Preprocessing_Files/columns.pkl", 'rb') as f:
    all_features_credit_card_fraud = pickle.load(f)
with open("Preprocessing File/ML-Project-10-Credit_Card_Fraud_Detection_Preprocessing_Files/scaler.pkl", 'rb') as f:
    scalers_credit_card_fraud = pickle.load(f)
with open("Best Features/ML-Project-10-Credit_Card_Fraud_Detection_Best_Features/best_features_svc.json", 'r') as file:
    best_features_svc_credit_card_fraud = json.load(file)
with open("Best Features/ML-Project-10-Credit_Card_Fraud_Detection_Best_Features/best_features_xgb.json", 'r') as file:
    best_features_xgb_credit_card_fraud = json.load(file)
with open("Best Features/ML-Project-10-Credit_Card_Fraud_Detection_Best_Features/best_features_rfc.json", 'r') as file:
    best_features_rfc_credit_card_fraud = json.load(file)
with open("Models/ML-Project-10-Credit_Card_Fraud_Detection_Models/credit_card_fraud_trained_svc_model.sav", 'rb') as f:
    loaded_model_svc_credit_card_fraud = pickle.load(f)
with open("Models/ML-Project-10-Credit_Card_Fraud_Detection_Models/credit_card_fraud_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_credit_card_fraud = pickle.load(f)
with open("Models/ML-Project-10-Credit_Card_Fraud_Detection_Models/credit_card_fraud_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_credit_card_fraud = pickle.load(f)


#loading the saved model of titanic survival prediction
with open("Preprocessing File/ML-Project-15-Titanic_Survival_Prediction_Preprocessing_Files/columns.pkl", 'rb') as f:
    all_columns_titanic_survival = pickle.load(f)
with open("Preprocessing File/ML-Project-15-Titanic_Survival_Prediction_Preprocessing_Files/cat_columns.pkl", 'rb') as f:
    cat_columns_titanic_survival = pickle.load(f)
with open("Preprocessing File/ML-Project-15-Titanic_Survival_Prediction_Preprocessing_Files/encoder.pkl", 'rb') as f:
    encoder_titanic_survival = pickle.load(f)
with open("Preprocessing File/ML-Project-15-Titanic_Survival_Prediction_Preprocessing_Files/encoded_columns.pkl", 'rb') as f:
    encoded_columns_titanic_survival = pickle.load(f)
with open("Preprocessing File/ML-Project-15-Titanic_Survival_Prediction_Preprocessing_Files/training_columns.pkl", 'rb') as f:
    training_columns_titanic_survival = pickle.load(f)
with open("Preprocessing File/ML-Project-15-Titanic_Survival_Prediction_Preprocessing_Files/scaler.pkl", 'rb') as f:
    scaler_titanic_survival = pickle.load(f)
with open("Best Features/ML-Project-15-Titanic_Survival_Prediction_Best_Features/best_features_knn.json", 'r') as file:
    best_features_knn_titanic_survival = json.load(file)
with open("Best Features/ML-Project-15-Titanic_Survival_Prediction_Best_Features/best_features_xgb.json", 'r') as file:
    best_features_xgb_titanic_survival = json.load(file)
with open("Best Features/ML-Project-15-Titanic_Survival_Prediction_Best_Features/best_features_lr.json", 'r') as file:
    best_features_lr_titanic_survival = json.load(file)
with open("Models/ML-Project-15-Titanic_Survival_Prediction_Models/titanic_survival_trained_knn_model.sav", 'rb') as f:
    loaded_model_knn_titanic_survival = pickle.load(f)
with open("Models/ML-Project-15-Titanic_Survival_Prediction_Models/titanic_survival_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_titanic_survival = pickle.load(f)
with open("Models/ML-Project-15-Titanic_Survival_Prediction_Models/titanic_survival_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_titanic_survival = pickle.load(f)


#loading the saved model of spam mail prediction
with open("Preprocessing File/ML-Project-17-Spam_Mail_Prediction_Preprocessing_Files/feature_extraction.pkl", 'rb') as f:
    loaded_model_feature_extraction_spam_mail = pickle.load(f)
with open("Models/ML-Project-17-Spam_Mail_Prediction_Models/spam_mail_prediction_trained_svc_model.sav", 'rb') as f:
    loaded_model_svc_spam_mail = pickle.load(f)
with open("Models/ML-Project-17-Spam_Mail_Prediction_Models/spam_mail_prediction_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_spam_mail = pickle.load(f)
with open("Models/ML-Project-17-Spam_Mail_Prediction_Models/spam_mail_prediction_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_spam_mail = pickle.load(f)


def loan_status_prediction(input_data):

    #loading columns
    columns = all_columns_loan_status
    
    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=columns)
    
    # Convert the categorical columns to string type
    df[cat_columns_loan_status] = df[cat_columns_loan_status].astype('str')
    
    # Encode the categorical columns
    input_data_encoded = encoder_loan_status.transform(df[cat_columns_loan_status])
    
    # Create a DataFrame with the encoded features
    input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_columns_loan_status)
    
    # Add the remaining non-categorical columns
    input_data_final_encoded = pd.concat([df.drop(cat_columns_loan_status, axis=1).reset_index(drop=True), input_data_encoded_df], axis=1)
    
    # Standardize the input data
    input_data_scaled = scaler_loan_status.transform(input_data_final_encoded)
    
    # Create a DataFrame with the standardized features
    input_data_df = pd.DataFrame(input_data_scaled, columns=training_columns_loan_status)
    
    #loading best features
    df_best_features_xgb = input_data_df[best_features_xgb_loan_status]
    df_best_features_rfc = input_data_df[best_features_rfc_loan_status]
    df_best_features_svc = input_data_df[best_features_svc_loan_status]
    
    #predictions
    prediction1_loan_status = loaded_model_xgb_loan_status.predict(df_best_features_xgb)
    prediction2_loan_status = loaded_model_rfc_loan_status.predict(df_best_features_rfc)
    prediction3_loan_status = loaded_model_svc_loan_status.predict(df_best_features_svc)
    
    return prediction1_loan_status , prediction2_loan_status, prediction3_loan_status


def wine_quality_prediction(input_data):

    df = pd.DataFrame([input_data], columns=all_features_wine_quality)

    df[all_features_wine_quality] = scalers_wine_quality.transform(df[all_features_wine_quality])
    
    df_best_features_rfc = df[best_features_rfc_wine_quality]
    df_best_features_xgb = df[best_features_xgb_wine_quality]
    df_best_features_knn = df[best_features_knn_wine_quality]
    
    prediction1_wine_quality = loaded_model_rfc_wine_quality.predict(df_best_features_rfc)
    prediction2_wine_quality = loaded_model_xgb_wine_quality.predict(df_best_features_xgb)
    prediction3_wine_quality = loaded_model_knn_wine_quality.predict(df_best_features_knn)
    
    prediction1_wine_quality = prediction1_wine_quality+3
    prediction2_wine_quality = prediction2_wine_quality+3
    prediction3_wine_quality = prediction3_wine_quality+3
    return prediction1_wine_quality , prediction2_wine_quality, prediction3_wine_quality


def credit_card_fraud_detection(input_data):

    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=all_features_credit_card_fraud)
    
    #loading columns
    df[all_features_credit_card_fraud] = scalers_credit_card_fraud.transform(df[all_features_credit_card_fraud])
    
    #loading best features
    df_best_features_svc = df[best_features_svc_credit_card_fraud]
    df_best_features_xgb = df[best_features_xgb_credit_card_fraud]
    df_best_features_rfc = df[best_features_rfc_credit_card_fraud]
    
    #predictions
    prediction1_credit_card_fraud = loaded_model_svc_credit_card_fraud.predict(df_best_features_svc)
    prediction2_credit_card_fraud = loaded_model_xgb_credit_card_fraud.predict(df_best_features_xgb)
    prediction3_credit_card_fraud = loaded_model_rfc_credit_card_fraud.predict(df_best_features_rfc)
  
    return prediction1_credit_card_fraud , prediction2_credit_card_fraud, prediction3_credit_card_fraud


def titanic_survival_prediction(input_data):

    #loading columns
    columns = all_columns_titanic_survival
    
    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=columns)
    
    # Convert the categorical columns to string type
    df[cat_columns_titanic_survival] = df[cat_columns_titanic_survival].astype('str')
    
    # Encode the categorical columns
    input_data_encoded = encoder_titanic_survival.transform(df[cat_columns_titanic_survival])
    
    # Create a DataFrame with the encoded features
    input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_columns_titanic_survival)
    
    # Add the remaining non-categorical columns
    input_data_final_encoded = pd.concat([df.drop(cat_columns_titanic_survival, axis=1).reset_index(drop=True), input_data_encoded_df], axis=1)
    
    # Standardize the input data
    input_data_scaled = scaler_titanic_survival.transform(input_data_final_encoded)
    
    # Create a DataFrame with the standardized features
    input_data_df = pd.DataFrame(input_data_scaled, columns=training_columns_titanic_survival)
    
    #loading best features
    df_best_features_knn = input_data_df[best_features_knn_titanic_survival]
    df_best_features_xgb = input_data_df[best_features_xgb_titanic_survival]
    df_best_features_lr = input_data_df[best_features_lr_titanic_survival]
    
    #predictions
    prediction1_titanic_survival = loaded_model_knn_titanic_survival.predict(df_best_features_knn)
    prediction2_titanic_survival = loaded_model_xgb_titanic_survival.predict(df_best_features_xgb)
    prediction3_titanic_survival = loaded_model_lr_titanic_survival.predict(df_best_features_lr)
  
    return prediction1_titanic_survival , prediction2_titanic_survival, prediction3_titanic_survival

def spam_mail_prediction(input_mail):

    #convert text to feature vectors
    input_data_features = loaded_model_feature_extraction_spam_mail.transform(input_mail)
    
    #predictions
    prediction1_spam_mail = loaded_model_svc_spam_mail.predict(input_data_features)
    prediction2_spam_mail = loaded_model_xgb_spam_mail.predict(input_data_features)
    prediction3_spam_mail = loaded_model_rfc_spam_mail.predict(input_data_features)
    
    return prediction1_spam_mail , prediction2_spam_mail, prediction3_spam_mail


def main():
    # sidebar for navigate

    with st.sidebar:
    
        selected = option_menu('ML Classification Projects WebApp System',
                           
                            ['Spam Mail Prediction',
                            'Loan Status Prediction',
                            'Titanic Survival Prediction',
                            'Wine Quality Prediction',
                            'Credit Card Prediction'],
                           
                           icons = ['envelope-slash','bank','water','cup-straw','credit-card'],
                           
                           default_index = 0)

    # Spam Mail Prediction Page
    if( selected == 'Spam Mail Prediction'):
        
        #giving a title
        st.title('Spam Mail Prediction Web App')
        
        #getting input data from user
        
        mail = st.text_input("Enter Mail Here")
        
        # code for prediction
        predict_mail_svc = ''
        predict_mail_xgb = ''
        predict_mail_rfc = ''
        
        predict_mail_svc,predict_mail_xgb,predict_mail_rfc = spam_mail_prediction([mail])
        
        #creating a button for Prediction
        if st.button("Predict Mail"):
            if(predict_mail_svc[0]==0):
                prediction = 'Ham Mail' 
            else:
                prediction = 'Spam Mail'
            st.write(f"Prediction: {prediction}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Mail with Support Vector Classifier"):
                if(predict_mail_svc[0]==0):
                    prediction = 'Ham Mail' 
                else:
                    prediction = 'Spam Mail'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Mail with XG Boost Classifier"):
                if(predict_mail_xgb[0]==0):
                    prediction = 'Ham Mail' 
                else:
                    prediction = 'Spam Mail'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Mail with Random Forest Classifier"):
                if(predict_mail_rfc[0]==0):
                    prediction = 'Ham Mail' 
                else:
                    prediction = 'Spam Mail'
                st.write(f"Prediction: {prediction}")
    
 
    # Loan Status Prediction Page
    if( selected == 'Loan Status Prediction'):
        
        
        #giving a title
        st.title('Loan Status Prediction using ML')
        
        col1 , col2 = st.columns(2)
        #getting input data from user
        with col1:
            no_of_dependents = st.number_input("No of Dependents")
        with col2:
            education = st.selectbox('Education',('Graduate', 'Not Graduate')) 
        with col1:
            self_employed = st.selectbox('Self_Employed',('Yes', 'No')) 
        with col2:
            income_annum = st.number_input("Income Annum")
        with col1:
            loan_amount = st.number_input("Loan Amount")
        with col2:
            loan_term = st.number_input("Loan Term")
        with col1:
            cibil_score = st.number_input('Cibil Score')
        with col2:
            residential_assets_value = st.number_input('Residential Assets Value')
        with col1:
            commercial_assets_value = st.number_input('Commercial Assets Value')
        with col2:
            luxury_assets_value = st.number_input('Luxury Assets Value')
        with col1:
            bank_asset_value = st.number_input('Bank Asset Value')
            
            
        # code for prediction
        loan_status_xgb = ''
        loan_status_rfc = ''
        loan_status_svc = ''
        
        loan_status_xgb,loan_status_rfc,loan_status_svc = loan_status_prediction([no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value])
        
        
        #creating a button for Prediction
        if st.button("Predict Loan Status"):
            if(loan_status_xgb[0]==0):
                prediction = 'The Loan of the Person is Accepted' 
            else:
                prediction = 'The Loan of the Person is Rejected'
            st.write(f"Prediction: {prediction}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Loan Status with XG Boost Classifier"):
                if(loan_status_xgb[0]==0):
                    prediction = 'The Loan of the Person is Accepted' 
                else:
                    prediction = 'The Loan of the Person is Rejected'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Loan Status with Random Forest Classifier"):
                if(loan_status_rfc[0]==0):
                    prediction = 'The Loan of the Person is Accepted' 
                else:
                    prediction = 'The Loan of the Person is Rejected'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Loan Status with Support Vector Classifier"):
                if(loan_status_svc[0]==0):
                    prediction = 'The Loan of the Person is Accepted' 
                else:
                    prediction = 'The Loan of the Person is Rejected'
                st.write(f"Prediction: {prediction}")  

    # Wine Quality Prediction Page
    if( selected == 'Wine Quality Prediction'):
        
        #giving a title
        st.title('Wine Quality Prediction Web App')
        
        col1 , col2 = st.columns(2)
        #getting input data from user
        with col1:
            fixed_acidity = st.number_input("Fixed Acidity of Wine")
        with col2:
            volatile_acidity = st.number_input("Volatile Acidity of Wine",format="%.3f")
        with col1:
            citric_acid = st.number_input("Citric Acid in Wine")
        with col2:
            residual_sugar = st.number_input("Residual Sugar in Wine")
        with col1:
            chlorides = st.number_input("Chlorides in Wine",format="%.4f")
        with col2:
            free_sulfur_dioxide = st.number_input("Free Sulfur-dioxide in Wine")
        with col1:
            total_sulfur_dioxide = st.number_input("Total Sulfur-dioxide in Wine")
        with col2:
            density = st.number_input("Density of Wine",format="%.4f")
        with col1:
            pH = st.number_input("pH of Wine")
        with col2:
            sulphates = st.number_input("Sulphates in Wine")
        with col1:
            alcohol = st.number_input("Alcohol in Wine")
       
        # code for prediction
        wine_quality_rfc = ''
        wine_quality_xgb = ''
        wine_quality_knn = ''
        
        wine_quality_rfc,wine_quality_xgb,wine_quality_knn = wine_quality_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
        
        #creating a button for Prediction
        if st.button("Predict Wine Quality"):
            st.write(f"Prediction(Range = 3 to 8): {wine_quality_rfc[0]}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Diabetes with Random Forest Classifier"):
                st.write(f"Quality of wine (Range : 3 to 8) = {wine_quality_rfc[0]}")
            if st.button("Predict Diabetes with XG Boost Classifier"):
                st.write(f"Quality of wine (Range : 3 to 8) = {wine_quality_xgb[0]}")
            if st.button("Predict Diabetes with K Nearest Neighbours"):
                st.write(f"Quality of wine (Range : 3 to 8) = {wine_quality_knn[0]}") 
        
    
    # Credit Card Prediction Page
    if( selected == 'Credit Card Prediction'):
        
        #giving a title
        st.title('Credit Card Fraud Detection Web App')
        
        col1 ,col2 ,col3 = st.columns(3)
        #getting input data from user
        with col1:
            v1 = st.number_input("V1",format="%.8f")
        with col2:
            v2 = st.number_input("V2",format="%.8f")
        with col3:	
            v3 = st.number_input("V3",format="%.8f") 
        with col1:
            v4 = st.number_input("V4",format="%.8f") 
        with col2:
            v5 = st.number_input("V5",format="%.8f")
        with col3:
            v6 = st.number_input("V6",format="%.8f")
        with col1:
            v7 = st.number_input("V7",format="%.8f")
        with col2:
            v8 = st.number_input("V8",format="%.8f")
        with col3:
            v9 = st.number_input("V9",format="%.8f")
        with col1:
            v10 = st.number_input("V10",format="%.8f")
        with col2:
            v11 = st.number_input("V11",format="%.8f")
        with col3:
            v12 = st.number_input("V12",format="%.8f")
        with col1:
            v13 = st.number_input("V13",format="%.8f")
        with col2:
            v14 = st.number_input("V14",format="%.8f")
        with col3:
            v15 = st.number_input("V15",format="%.8f")
        with col1:
            v16 = st.number_input("V16",format="%.8f")
        with col2:
            v17 = st.number_input("V17",format="%.8f")
        with col3:
            v18 = st.number_input("V18",format="%.8f")
        with col1:
            v19 = st.number_input("V19",format="%.8f")
        with col2:
            v20 = st.number_input("V20",format="%.8f")
        with col3:
            v21 = st.number_input("V21",format="%.8f")
        with col1:
            v22 = st.number_input("V22",format="%.8f")
        with col2:
            v23 = st.number_input("V23",format="%.8f")
        with col3:
            v24 = st.number_input("V24",format="%.8f")
        with col1:
            v25 = st.number_input("V25",format="%.8f")
        with col2:
            v26 = st.number_input("V26",format="%.8f")
        with col3:
            v27 = st.number_input("V27",format="%.8f")
        with col1:
            v28 = st.number_input("V28",format="%.8f")
        with col2:
            amount = st.number_input("Amount",format="%.4f")
        
        
        # code for prediction
        credit_card_fraud_svc = ''
        credit_card_fraud_xgb = ''
        credit_card_fraud_rfc = ''
        credit_card_fraud_svc,credit_card_fraud_xgb,credit_card_fraud_rfc =credit_card_fraud_detection([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,amount])
        
        
        #creating a button for Prediction
        if st.button("Predict Credit Card Transaction"):
            if(credit_card_fraud_xgb[0]==0):
                prediction = 'The Transaction is Legit' 
            else:
                prediction = 'The Transaction is Fraud'
            st.write(f"Prediction: {prediction}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Credit Card Transaction with XG Boost Classifier"):
                if(credit_card_fraud_xgb[0]==0):
                    prediction = 'The Transaction is Legit' 
                else:
                    prediction = 'The Transaction is Fraud'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Credit Card Transaction with Support Vector Classifer"):
                if(credit_card_fraud_svc[0]==0):
                    prediction = 'The Transaction is Legit' 
                else:
                    prediction = 'The Transaction is Fraud'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Credit Card Transaction with Random Forest Classifier"):
                if(credit_card_fraud_rfc[0]==0):
                    prediction = 'The Transaction is Legit' 
                else:
                    prediction = 'The Transaction is Fraud'
                st.write(f"Prediction: {prediction}")

     # Titanic Survival Prediction Page
    if( selected == 'Titanic Survival Prediction'):
         
        #giving a title
        st.title('Titanic Survival Prediction Web App')
        
        #getting input data from user
        
        option1 = st.selectbox('Passenger class',('First class', 'Second class','Third class'))
        if option1 == 'First class':
            pclass = 1
        elif option1 == 'Second class':
            pclass = 2
        else:
            pclass = 3
        
        sex = st.selectbox('Gender',('male', 'female'))
        
        age = st.number_input("Age of the person",format="%.0f")
        
        sibsp = st.number_input("No of Siblings",format="%.0f")
        
        parch = st.number_input("Parch")
        
        fare = st.number_input("Fare")
        
        option3 = st.selectbox('Embarked',('Southampton','Cherbourg', 'Queenstown'))
        if option3 == 'Southampton':
            embarked = 'S'
        elif option3 == 'Cherbourg':
            embarked = 'C'
        else:
            embarked = 'Q'
        
        # code for prediction
        person_survived_knn = ''
        person_survived_xgb = ''
        person_survived_lr = ''
        
        
        person_survived_knn,person_survived_xgb,person_survived_lr = titanic_survival_prediction([pclass,sex,age,sibsp,parch,fare,embarked])
        
        
        #creating a button for Prediction
        if st.button("Predict Survival"):
            if(person_survived_lr[0]==0):
                prediction = 'The Person does not Survived' 
            else:
                prediction = 'The Person Survived'
            st.write(f"Prediction: {prediction}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Survival with Logistics Regression"):
                if(person_survived_lr[0]==0):
                    prediction = 'The Person does not Survived' 
                else:
                    prediction = 'The Person Survived'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Survival with K Nearest Neighbour"):
                if(person_survived_knn[0]==0):
                    prediction = 'The Person does not Survived' 
                else:
                    prediction = 'The Person Survived'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Survival with XG Boost Classifier"):
                if(person_survived_xgb[0]==0):
                    prediction = 'The Person does not Survived' 
                else:
                    prediction = 'The Person Survived'
                st.write(f"Prediction: {prediction}")
    
if __name__ == '__main__':
    main()





