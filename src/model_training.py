import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import utils
from sklearn.impute import SimpleImputer
from visualization_and_analysis import create_matplotlib_table, plot_confusion_matrix
import os
from imblearn.over_sampling import RandomOverSampler

# Get X (features) and y (target)
def get_X_y(data):
    X = data.drop(['veracity', 'nationality', 'position', 'source'], axis=1)
    y = data['veracity']
    return X, y

# Impute missing values using mean imputation
def impute_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    return X_imputed

def apply_ros(X_train, y_train):
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def split_data(data):
    X, y = get_X_y(data)
    X_imputed = impute_missing_values(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.4, random_state=42)
    X_train_resampled, y_train_resampled = apply_ros(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def top_5_feature_importances(model, X_train):
    total_features = len(X_train.columns)
    feature_importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    top_5_features = feature_importances[:5]

    return total_features, top_5_features

def convert_top_5_feature_importances_to_df(top_5_features):
    return pd.DataFrame(top_5_features, columns=['Feature', 'Importance']).set_index('Feature')
    
def print_top_5_feature_importances(model_name, model, X_train):
    total_features, top_5_features = top_5_feature_importances(model, X_train)
    print(f"\n{model_name} top 5 features out of {total_features} features:")
    for feature, importance in top_5_features:
        print(f"{feature}: {importance:.4f}")

def convert_model_report_to_df(report):
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=['support'])
    report_df = report_df.drop(['macro avg', 'weighted avg'])
    return report_df

def print_model_results(model_name, accuracy, report_df):
    print(f"{model_name} accuracy: {accuracy:.4f}")
    print("Classification report table:")
    print(report_df)

def print_confusion_matrix(model_name, y_test, y_pred):
    print(f"{model_name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def train_and_evaluate_models(data):
    X_train, X_test, y_train, y_test = split_data(data)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss')
    }

    # Using StratifiedKFold to maintain class distribution across folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        # Perform k-fold cross-validation and compute mean accuracy
        cv_accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))
        print(f"{model_name} cross-validated accuracy: {cv_accuracy}")

        # Train and evaluate the model on the test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Add cross-validated accuracy to the report dictionary
        report['cross_validated_accuracy'] = cv_accuracy

        # Remove support column and Macro average and Weighted average rows and convertq to dataframe
        report_df = convert_model_report_to_df(report).round(4)
        print_model_results(model_name, accuracy, report_df)
        report_save_path = os.path.join('results', f'{model_name.lower()}_classification_report.png')
        create_matplotlib_table(report_df, report_save_path)

        print_top_5_feature_importances(model_name, model, X_train)

        top_5_features = top_5_feature_importances(model, X_train)[1]
        top_5_importances_df = convert_top_5_feature_importances_to_df(top_5_features)
        top_5_importances_save_path = os.path.join('results', f'{model_name.lower()}_top_5_features.png')
        create_matplotlib_table(top_5_importances_df, top_5_importances_save_path)
        

        print_confusion_matrix(model_name, y_test, y_pred)
        confusion_save_path = os.path.join('results', f'{model_name.lower()}_confusion_matrix.png')
        plot_confusion_matrix(model_name, y_test, y_pred, confusion_save_path)

def main():
    output_data = utils.pandas_load_csv("output_data.csv")
    
    train_and_evaluate_models(output_data)

if __name__ == '__main__':
    main()