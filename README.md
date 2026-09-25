# 🩺 Diabetes Prediction Using Machine Learning

A machine learning project that predicts whether a person is likely to have diabetes based on diagnostic and health-related features.

The project covers the complete machine learning workflow — from data loading and exploratory data analysis to model training, evaluation, model serialization, and deployment using Streamlit.

## 🚀 Live Demo

The trained model is deployed as an interactive Streamlit web application.

👉 Live Demo: https://diabetes-prediction0.streamlit.app/

---

## 📌 Project Overview

Diabetes is a common health condition that can be influenced by several factors such as glucose level, blood pressure, BMI, age, insulin level, and family history.

The objective of this project is to build a classification model that can analyze these input features and predict the corresponding diabetes outcome.

This project demonstrates an end-to-end machine learning workflow using Python and Scikit-learn, with a Streamlit interface for interactive predictions.

---

## 🎯 Project Objectives

- Load and understand the diabetes dataset
- Perform data quality checks
- Explore the dataset using EDA
- Prepare the data for machine learning
- Split the dataset into training and testing sets
- Apply feature scaling where required
- Train and compare multiple classification algorithms
- Evaluate model performance using classification metrics
- Save the trained model for deployment
- Build an interactive Streamlit application
- Deploy the application for real-time predictions

---

## 📊 Dataset

The project uses a diabetes dataset containing **768 records and 9 columns**.

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree/family-history related measure |
| Age | Age of the individual |
| Outcome | Target variable |

### Target Variable

`Outcome`

- `0` → No diabetes indicated by the dataset
- `1` → Diabetes indicated by the dataset

> **Note:** This is a machine learning project for educational and demonstration purposes. The prediction should not be considered a medical diagnosis.

---

## 🧠 Machine Learning Workflow

The project follows an end-to-end ML pipeline:

```text
Dataset
   ↓
Data Loading
   ↓
Data Quality Checks
   ↓
Exploratory Data Analysis
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
