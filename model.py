
import pickle

import pandas as pd     # Data preprocessiong operations
import inspect
import numpy as np     # Linear algebra operations
import csv     # CSV file operations 
import matplotlib.pyplot as plt     # Visualizations 
import seaborn as sns     # Visualizations 
import scipy.stats as stats 
from sklearn.preprocessing import LabelEncoder     # Encoding ordinal features
from sklearn.preprocessing import OneHotEncoder     # Encoding nominal features
from sklearn.feature_selection import SelectKBest, chi2     # Feature selection fun  ctions
from sklearn.preprocessing import StandardScaler     # Standrization
from sklearn.linear_model import LogisticRegression     # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier, StackingClassifier     # Random forest model and StackingClassifier lib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score    # Model evaluation metrics
from sklearn.model_selection import GridSearchCV     # Model hyperparameters grid
import optuna     # Hyperparameters fine-tuning
import logging     # Customizing fetch messages
from imblearn.over_sampling import SMOTE     # Handle dataset imbalance 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
import xgboost as xgb
import lightgbm as lgb

#Importing datasets
training_dataset = pd.read_csv("training_dataset.csv")
test_dataset = pd.read_csv("test_dataset.csv")

# Handeling outliers

def compute_bounds(df, column):
    '''
    Helper function that calculates the bounds of a column
    
    df :               Dataframe name
    Q1 :               First quantile
    Q3 :               Third quantile
    IQR :              Quantile range
    lower_bound :      Floor of values
    upper_bound :      Ceil of values
    '''
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return lower_bound, upper_bound

def Replace_outliers_with_mean(df, column):
    '''
    Checks for outliers in a column and replaces them with the mean value
    
    df :               Dataframe name
    column:            Column name
    mean_value :       Mean value of the column
    '''
    
    lower_bound, upper_bound = compute_bounds(df, column)
    
    mean_value = df[column].mean()
    
    df[column] = df[column].where((df[column] >= lower_bound) & (df[column] <= upper_bound), mean_value)
    
    return df

def Replace_outliers_with_median(df, column):
    '''
    Checks for outliers in a column and replaces them with the median value
    
    df :               Dataframe name
    column:            Column name
    mean_value :       Median value of the column
    '''
    
    lower_bound, upper_bound = compute_bounds(df, column)
    
    median_value = df[column].median()
    
    df[column] = df[column].where((df[column] >= lower_bound) & (df[column] <= upper_bound), median_value)
    
    return df

def handle_outliers(df, median_cols, mean_cols):
    '''
    Changes outliers in the dataset using the adequat function
    
    df :                  Dataframe name
    median_cols :         Columns to replace their outliers with median
    mean_cols :           Columns to replace their outliers with mean
    '''
    for i in range(2): 
        for column in mean_cols:
            df = Replace_outliers_with_mean(df, column)
            print(f"mean value after iteration {i+1} of {column} : {df[column].mean()}")
        for column in median_cols:
            df = Replace_outliers_with_median(df, column)
            print(f"median value after iteration {i+1} of {column} : {df[column].mean()}")
    
    # Deleting rows with very large outliers        
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        lower_bound, upper_bound = compute_bounds(df, column)
        # Identify and drop rows with outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        df = df[~outliers]

replace_outliers_with_mean_columns = ['Total day minutes', 
                                 'Total day charge', 
                                 'Total eve minutes', 
                                 'Total eve charge', 
                                 'Total night minutes', 
                                 'Total night charge', 
                                 'Total intl minutes', 
                                 'Total intl charge']

replace_outliers_with_median_columns = ['Account length',
                                   'Total day calls',
                                   'Total eve calls',
                                   'Total night calls',
                                   'Total intl calls',
                                   'Customer service calls']


handle_outliers(training_dataset, replace_outliers_with_median_columns, replace_outliers_with_mean_columns)
handle_outliers(test_dataset, replace_outliers_with_median_columns, replace_outliers_with_mean_columns)

# Encoding
def encoding_categorical_features(df, columns):
    '''
    Encodes ordinal categorical features
    
    df :                   Dataframe name
    columns :              Columns list
    label_encoder :        LabelEncoder instance
    '''
    label_encoder = LabelEncoder()
    for column in columns:
        # Encoding the categorical column
        df[column] = label_encoder.fit_transform(df[column])

categorical_columns = ['State', 'International plan', 'Voice mail plan', 'Churn']

encoding_categorical_features(training_dataset, categorical_columns)
encoding_categorical_features(test_dataset, categorical_columns)
def delete_correlated_features(df, threshold):
    '''
    Deletes features with corellation higher than the threshold
    
    df :                          Original dataframe name
    threshold :                   Correlation threshold
    correlation_matrix :          Df correlation matrix
    upper_triangle :              Upper triangle of the correlation matrix
    to_drop :                     Columns to drop
    '''
    correlation_matrix = df.corr()

    # Find the upper triangle of the correlation matrix (we only need to check one half)
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Get a list of columns with correlations above the threshold
    to_drop = [column for column in upper_triangle.columns if any(abs(upper_triangle[column]) > threshold)]

    # Drop highly correlated features
    df = df.drop(columns=to_drop)
    
    return df

def select_best_features(X, y, nbr_features):
    '''
    Selects the best features for the modeling phase
    
    X :                        Features list
    y :                        Target
    nbr_features :             Number of features to select
    test :                
    fit :                      Feature selection fit model
    selected_features :        New features array
    '''
    test = SelectKBest(score_func=chi2, k=nbr_features)
    fit = test.fit(X, y)
    
    selected_features = X.columns[test.get_support()]
    
    X = fit.transform(X)
    
    return X, selected_features
# Splitting the features from the tagrget
X1 = training_dataset.drop(columns=['Churn'])  # All columns except target
y1 = training_dataset['Churn']  # The target variable
X2 = test_dataset.drop(columns=['Churn'])  
y2 = test_dataset['Churn']

X1 = delete_correlated_features(X1, threshold=0.9)
X2 = delete_correlated_features(X2, threshold=0.9)

# Selecting best features
X1, selected_features = select_best_features(X1, y1, nbr_features=6)
X2, selected_features = select_best_features(X2, y2, nbr_features=6)

# Standarization
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
X2 = scaler.transform(X2)

def evalution_base(X, y, y_pred):
    '''
    Evaluates model performance
    
    accuracy :        Model accuracy
    class_report:     Model classification report
    '''
    accuracy = accuracy_score(y, y_pred)
    class_report = classification_report(y, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"classification report:\n{class_report}")
    
    return accuracy, class_report

# XGBoost model

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
def xgboost_with_optuna(X, y, X_test, y_test):
    """
    Fine-tunes an XGBoost model using Optuna for hyperparameter optimization.
    
    X :               Features for training.
    y :               Target for training.
    X_test :          Features for testing.
    y_test :          Target for testing.
    best_model:       The trained XGBoost model with the best parameters.
    y_pred:           Predictions made by the best model on X_test.
    accuracy:         Accuracy score of the best model on the test set.
    class_report:     Classification report of the best model.
    """
    
    # Set the logging level to suppress info and warnings
    logging.basicConfig(level=logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """
        Objective function for Optuna to optimize XGBoost hyperparameters.
        """
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
        max_depth = trial.suggest_int('max_depth', 4, 12)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        gamma = trial.suggest_float('gamma', 0.0, 5.0)
        scale_pos_weight = trial.suggest_float('scale_pos_weight', 1.0, 5.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
        max_delta_step = trial.suggest_int('max_delta_step', 0, 10)
        tree_method = trial.suggest_categorical('tree_method', ['auto', 'hist', 'gpu_hist'])
        grow_policy = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

        try:
            model = XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                max_delta_step=max_delta_step,
                tree_method=tree_method,
                grow_policy=grow_policy,
                random_state=42
            )
            
            # Fit model without eval_metric
            model.fit(
                X, y,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        except ValueError:
            # Handle invalid combinations
            return float('inf')

        return -accuracy  # Maximize accuracy by minimizing its negative

    # Create and optimize the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # Increase trials for better exploration

    # Retrieve the best hyperparameters
    best_params = study.best_params

    # Train the final model using the best parameters
    best_model = XGBClassifier(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        scale_pos_weight=best_params['scale_pos_weight'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        max_delta_step=best_params['max_delta_step'],
        tree_method=best_params['tree_method'],
        grow_policy=best_params['grow_policy'],
        random_state=42
    )
    
    # Final model fitting (no early stopping)
    best_model.fit(
        X, y,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluating performance
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Model Hyperparameters : {best_params}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")

    return best_model, y_pred, accuracy, class_report

xgbm_bestmodel, xgbm_bestpredictions, xgbm_bestacc, xgbm_bestreport = xgboost_with_optuna(X1, y1, X2, y2)

#Pickeling the model
with open('model.pkl', 'wb') as f:
    pickle.dump(xgbm_bestmodel, f)