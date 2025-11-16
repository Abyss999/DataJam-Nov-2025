# DataJam-Nov-2025

## Overview
This repository contains the code and assets used for our **DataJam November 2025** project.  
The project focuses on training and evaluating multiple machine learning models for a classification task, including Logistic Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost, and LightGBM.  
The final model we used in our presentation was **LightGBM with RandomizedSearchCV**, which achieved the best overall performance.

## Repository Structure
```
DataJam-Nov-2025/
│
├── Dataset/ # Dataset files used for model training/testing
├── catboost_info/ # Auto-generated CatBoost metadata folder
│
├── Imports.py # Centralized imports + helper utilities
├── main.py # Main script that loads data, preprocesses, trains models, evaluates results
│
├── requirements.txt # Python dependencies for reproducing the environment
└── .gitignore # Ignored files for version control
```

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Abyss999/DataJam-Nov-2025.git
cd DataJam-Nov-2025
```

### 2. Virtual Environment 
```bash
python3 -m venv datajam
```
#### Mac/Linux 
```bash
source datajam/bin/activate
```
#### Windows 
```bash
datajam\Scripts\activate
```
## 3. Install
```bash
pip install -r requirements.txt
```
### 4. Run Project 
```
python main.py
```

## This script:
- loads and cleans the dataset
- encodes categorical features
- normalizes numerical features
- trains multiple ML models
- evaluates them using accuracy, classification reports, and confusion matrices
- compares performance across models

## Models Implemented

The following algorithms are implemented in main.py:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- LightGBM (best performing model with RandomizedSearchCV)


