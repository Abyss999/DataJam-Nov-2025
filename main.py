import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# loads the data 
df = pd.read_csv("Dataset/ai4i2020.csv")
df = df.drop(columns=["UDI", "Product ID"])

# separates features and targets 
X = df.drop(columns=["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]) # targets 
y = df["Machine failure"]

# encodes categorical features
encoder = LabelEncoder()
X["Type"] = encoder.fit_transform(X["Type"])

# scales the numeric features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# print(X_scaled)


# logistic regression model 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# creates and trains the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# makes predictions
y_pred = model.predict(X_test)

# evaluates the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(X.head())
print(y.head())

# make a decision tree and test it using stratified k folds 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# initializes the decision tree classifier
dt_model = DecisionTreeClassifier(random_state=42)

# initializes stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # trains the model
    dt_model.fit(X_train, y_train)
    # makes predictions
    y_pred = dt_model.predict(X_test)
    # calculates accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# prints the average accuracy
print(f"Average Accuracy from Stratified K-Fold: {np.mean(accuracies)}")    

# now do random forest and test it using stratified k folds 
from sklearn.ensemble import RandomForestClassifier


# initializes the random forest classifier
rf_model = RandomForestClassifier(random_state=42)
accuracies = []
for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # trains the model
    rf_model.fit(X_train, y_train)
    # makes predictions
    y_pred = rf_model.predict(X_test)
    # calculates accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)


# prints the average accuracy
print(f"Average Accuracy from Stratified K-Fold (Random Forest): {np.mean(accuracies)}")

# do boosting and test it using stratified k folds to find accuracy
from sklearn.ensemble import GradientBoostingClassifier

# initializes the gradient boosting classifier
gb_model = GradientBoostingClassifier(random_state=42)
accuracies = []
for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # trains the model
    gb_model.fit(X_train, y_train)
    # makes predictions
    y_pred = gb_model.predict(X_test)
    # calculates accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# prints the average accuracy
print(f"Average Accuracy from Stratified K-Fold (Gradient Boosting): {np.mean(accuracies)}") 

# do boosting and test it using stratified k folds to find accuracy
from sklearn.ensemble import GradientBoostingClassifier


# initializes the gradient boosting classifier
gb_model = GradientBoostingClassifier(random_state=42)
accuracies = []
for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # trains the model
    gb_model.fit(X_train, y_train)
    # makes predictions
    y_pred = gb_model.predict(X_test)
    # calculates accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# prints the average accuracy
print(f"Average Accuracy from Stratified K-Fold (Gradient Boosting): {np.mean(accuracies)}") 

# randomize search cv for gradient boosting classifier
from sklearn.model_selection import RandomizedSearchCV


# defines the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2
],

    'max_depth': [3, 5, 7, 9]
}

# initializes the randomized search
random_search = RandomizedSearchCV(
    estimator=gb_model,
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=skf,
    random_state=42,
    n_jobs=-1
)

# fits the randomized search
random_search.fit(X_scaled, y)

# prints the best parameters and best score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")

# final evaluation with best parameters
best_gb_model = random_search.best_estimator_
y_pred = best_gb_model.predict(X_scaled)
final_accuracy = accuracy_score(y, y_pred)
print(f"Final Accuracy with Best Parameters: {final_accuracy}")

# random forest classifier with randomized search cv
# defines the parameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# initializes the randomized search
random_search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid_rf,
    n_iter=10,
    scoring='accuracy',
    cv=skf,
    random_state=42,
    n_jobs=-1
)

# fits the randomized search
random_search_rf.fit(X_scaled, y)

# prints the best parameters and best score
print(f"Best Parameters (RF): {random_search_rf.best_params_}")
print(f"Best Score (RF): {random_search_rf.best_score_}")
# final evaluation with best parameters
best_rf_model = random_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_scaled)
final_accuracy_rf = accuracy_score(y, y_pred_rf)
print(f"Final Accuracy with Best Parameters (RF): {final_accuracy_rf}")

# now do xgboost classifier with randomized search cv
import xgboost as xgb


# initializes the xgboost classifier
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False,
                                eval_metric='logloss')

# defines the parameter grid
param_grid_xgb = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2
],

    'max_depth': [3, 5, 7, 9]
}

# initializes the randomized search
random_search_xgb = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid_xgb,
    n_iter=10,
    scoring='accuracy',
    cv=skf,
    random_state=42,
    n_jobs=-1
)

# fits the randomized search
random_search_xgb.fit(X_scaled, y)

# prints the best parameters and best score
print(f"Best Parameters (XGB): {random_search_xgb.best_params_}")
print(f"Best Score (XGB): {random_search_xgb.best_score_}")

# final evaluation with best parameters
best_xgb_model = random_search_xgb.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_scaled)
final_accuracy_xgb = accuracy_score(y, y_pred_xgb)
print(f"Final Accuracy with Best Parameters (XGB): {final_accuracy_xgb}")

# light gbm classifier with randomized search cv
import lightgbm as lgb


# initializes the light gbm classifier
lgb_model = lgb.LGBMClassifier(random_state=42)

# defines the parameter grid
param_grid_lgb = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2
],

    'max_depth': [3, 5, 7, 9]
}

# initializes the randomized search
random_search_lgb = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid_lgb,
    n_iter=10,
    scoring='accuracy',
    cv=skf,
    random_state=42,
    n_jobs=-1
)

# fits the randomized search
random_search_lgb.fit(X_scaled, y)

# prints the best parameters and best score
print(f"Best Parameters (LGB): {random_search_lgb.best_params_}")
print(f"Best Score (LGB): {random_search_lgb.best_score_}")

# final evaluation with best parameters
best_lgb_model = random_search_lgb.best_estimator_
y_pred_lgb = best_lgb_model.predict(X_scaled)
final_accuracy_lgb = accuracy_score(y, y_pred_lgb)
print(f"Final Accuracy with Best Parameters (LGB): {final_accuracy_lgb}")

# cat boost classifier with randomized search cv
from catboost import CatBoostClassifier


# initializes the cat boost classifier
cat_model = CatBoostClassifier(random_state=42, verbose=0)

# defines the parameter grid
param_grid_cat = {
    'iterations': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2
],

    'depth': [3, 5, 7, 9]
}

# initializes the randomized search
random_search_cat = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=param_grid_cat,
    n_iter=10,
    scoring='accuracy',
    cv=skf,
    random_state=42,
    n_jobs=-1
)

# fits the randomized search
random_search_cat.fit(X_scaled, y)

# prints the best parameters and best score
print(f"Best Parameters (CatBoost): {random_search_cat.best_params_}")
print(f"Best Score (CatBoost): {random_search_cat.best_score_}")

# final evaluation with best parameters
best_cat_model = random_search_cat.best_estimator_
y_pred_cat = best_cat_model.predict(X_scaled)
final_accuracy_cat = accuracy_score(y, y_pred_cat)
print(f"Final Accuracy with Best Parameters (CatBoost): {final_accuracy_cat}")
