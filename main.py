from Imports import *
from dataTransform import loader

class models():
    def __init__(self):
        self.X_scaled, self.y, self.X = loader()
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    def log_reg(self):
        # splits the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled,self.y, test_size=0.2, random_state=42)

        # creates and trains the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # makes predictions
        y_pred = model.predict(X_test)

        # evaluates the model
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(self.X.head())
        print(self.y.head())
    def decision_tree(self):
        dt_model = DecisionTreeClassifier(random_state=42)

        # initializes stratified k-fold cross-validation
        
        accuracies = []
        for train_index, test_index in self.skf.split(self.X_scaled, self.y):
            X_train, X_test = self.X_scaled[train_index], self.X_scaled[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            # trains the model
            dt_model.fit(X_train, y_train)
            # makes predictions
            y_pred = dt_model.predict(X_test)
            # calculates accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # prints the average accuracy
        print(f"Average Accuracy from Stratified K-Fold: {np.mean(accuracies)}")    
    def randomForest(self):
        rf_model = RandomForestClassifier(random_state=42)
        accuracies = []
        for train_index, test_index in self.skf.split(self.X_scaled, self.y):
            X_train, X_test = self.X_scaled[train_index], self.X_scaled[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            # trains the model
            rf_model.fit(X_train, y_train)
            # makes predictions
            y_pred = rf_model.predict(X_test)
            # calculates accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            print(f"Average Accuracy from Stratified K-Fold (Random Forest): {np.mean(accuracies)}")
    def randomForestRSCV(self):
        # defines the parameter grid
        param_grid_rf = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # initializes the randomized search
        random_search_rf = RandomizedSearchCV(
            estimator=self.rf_model,
            param_distributions=param_grid_rf,
            n_iter=10,
            scoring='accuracy',
            cv=self.skf,
            random_state=42,
            n_jobs=-1
        )

        # fits the randomized search
        random_search_rf.fit(self.X_scaled, self.y)

        # prints the best parameters and best score
        print(f"Best Parameters (RF): {random_search_rf.best_params_}")
        print(f"Best Score (RF): {random_search_rf.best_score_}")
        # final evaluation with best parameters
        best_rf_model = random_search_rf.best_estimator_
        y_pred_rf = best_rf_model.predict(self.X_scaled)
        final_accuracy_rf = accuracy_score(self.y, y_pred_rf)
        print(f"Final Accuracy with Best Parameters (RF): {final_accuracy_rf}")
    def gradientBoosting(self):
        gb_model = GradientBoostingClassifier(random_state=42)
        accuracies = []
        for train_index, test_index in self.skf.split(self.X_scaled, self.y):
            X_train, X_test = self.X_scaled[train_index], self.X_scaled[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            # trains the model
            gb_model.fit(X_train, y_train)
            # makes predictions
            y_pred = gb_model.predict(X_test)
            # calculates accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            # prints the average accuracy
            print(f"Average Accuracy from Stratified K-Fold (Gradient Boosting): {np.mean(self.accuracies)}") 
    def gradientBoostingRSCV(self):
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9]
        }
        # initializes the randomized search
        random_search = RandomizedSearchCV(
            estimator=self.gb_model,
            param_distributions=self.param_grid,
            n_iter=10,
            scoring='accuracy',
            cv=self.skf,
            random_state=42,
            n_jobs=-1
        )

        # fits the randomized search
        random_search.fit(self.X_scaled, self.y)

        # prints the best parameters and best score
        print(f"Best Parameters: {random_search.best_params_}")
        print(f"Best Score: {random_search.best_score_}")

        # final evaluation with best parameters
        best_gb_model = random_search.best_estimator_
        y_pred = best_gb_model.predict(self.X_scaled)
        final_accuracy = accuracy_score(self.y, y_pred)
        print(f"Final Accuracy with Best Parameters: {final_accuracy}")
    def lightGradientBoosting(self):
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
            cv=self.skf,
            random_state=42,
            n_jobs=-1
        )

        # fits the randomized search
        random_search_lgb.fit(self.X_scaled, self.y)

        # prints the best parameters and best score
        print(f"Best Parameters (LGB): {random_search_lgb.best_params_}")
        print(f"Best Score (LGB): {random_search_lgb.best_score_}")

        # final evaluation with best parameters
        best_lgb_model = random_search_lgb.best_estimator_
        y_pred_lgb = best_lgb_model.predict(self.X_scaled)
        final_accuracy_lgb = accuracy_score(self.y, y_pred_lgb)
        print(f"Final Accuracy with Best Parameters (LGB): {final_accuracy_lgb}")
    def catBoosting(self):
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
            cv=self.decision_treeskf,
            random_state=42,
            n_jobs=-1
        )

        # fits the randomized search
        random_search_cat.fit(self.X_scaled, self.y)

        # prints the best parameters and best score
        print(f"Best Parameters (CatBoost): {random_search_cat.best_params_}")
        print(f"Best Score (CatBoost): {random_search_cat.best_score_}")

        # final evaluation with best parameters
        best_cat_model = random_search_cat.best_estimator_
        y_pred_cat = best_cat_model.predict(self.X_scaled)
        final_accuracy_cat = accuracy_score(self.y, y_pred_cat)
        print(f"Final Accuracy with Best Parameters (CatBoost): {final_accuracy_cat}")
    def xgbBoosting(self):
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
            cv=self.skf,
            random_state=42,
            n_jobs=-1
        )

        # fits the randomized search
        random_search_xgb.fit(self.X_scaled, self.y)

        # prints the best parameters and best score
        print(f"Best Parameters (XGB): {random_search_xgb.best_params_}")
        print(f"Best Score (XGB): {random_search_xgb.best_score_}")

        # final evaluation with best parameters
        best_xgb_model = random_search_xgb.best_estimator_
        y_pred_xgb = best_xgb_model.predict(self.X_scaled)
        final_accuracy_xgb = accuracy_score(self.y, y_pred_xgb)
        print(f"Final Accuracy with Best Parameters (XGB): {final_accuracy_xgb}")

        
    """def fineTune(self):
        

        #
        # now do xgboost classifier with randomized search cv


        # initializes the xgboost classifier
        

        # light gbm classifier with randomized search cv



        

        # cat boost classifier with randomized search cv



        # initializes the cat boost classifier
        
"""



# Call the gradientBoosting method
models().lightGradientBoosting()