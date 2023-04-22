import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import utils
from sklearn.impute import SimpleImputer


# Load data from a CSV file
def load_data(filename):
    return pd.read_csv(utils.get_data_file_path(filename))

# Drop rows with missing veracity values
def drop_na_veracity(data):
    return data.dropna(subset=['veracity'])

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

def split_data(data):
    X, y = get_X_y(data)
    X_imputed = impute_missing_values(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def print_top_5_feature_importances(model_name, model, X_train):
    total_features = len(X_train.columns)
    print(f"\n{model_name} top 5 features out of {total_features} features:")
    feature_importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importances[:5]:
        print(f"{feature}: {importance}")

def print_model_results(model_name, accuracy, report):
    print(f"{model_name} accuracy: {accuracy}")
    # Remove support column and Macro average and Weighted average rows
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=['support'])
    report_df = report_df.drop(['macro avg', 'weighted avg'])
    
    # Round the classification report results
    report_df = report_df.round(decimals=2)
    
    print("Classification report table:")
    print(report_df)

def print_confusion_matrix(model_name, y_test, y_pred):
    print(f"{model_name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
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

        print_model_results(model_name, accuracy, report)
        print_top_5_feature_importances(model_name, model, X_train)
        print_confusion_matrix(model_name, y_test, y_pred)

if __name__ == '__main__':
    data = load_data("output_data.csv")
    X_train, X_test, y_train, y_test = split_data(data)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
