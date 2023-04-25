import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import utils
from sklearn.impute import SimpleImputer
from visualization_and_analysis import create_matplotlib_table, plot_confusion_matrix, plot_feature_importances
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

# Apply Random Over Sampling
def apply_ros(X_train, y_train):
    print("\nApplying Random Over Sampling...")
    print(f"Before applying ROS, the number of samples in the minority class: {y_train.value_counts()[0]}")
    print(f"Before applying ROS, the number of samples in the majority class: {y_train.value_counts()[1]}")

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    print(f"After applying ROS, the number of samples in the minority class: {y_train_resampled.value_counts()[0]}")
    print(f"After applying ROS, the number of samples in the majority class: {y_train_resampled.value_counts()[1]}")

    return X_train_resampled, y_train_resampled

# Split data into train and test sets. Oversamples the minority class if needed
def split_data(data, oversample=False):
    X, y = get_X_y(data)
    X_imputed = impute_missing_values(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.4, random_state=42)

    if oversample:
        X_train_resampled, y_train_resampled = apply_ros(X_train, y_train)
        return X_train_resampled, X_test, y_train_resampled, y_test
    else:
        return X_train, X_test, y_train, y_test

# Train and evaluate the model on the test set
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, cv):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Perform k-fold cross-validation and compute mean accuracy
    cv_accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))
    report = classification_report(y_test, y_pred, output_dict=True)
    # Add cross-validated accuracy to the report dictionary
    report['cross_validated_accuracy'] = cv_accuracy

    return accuracy, cv_accuracy, report, y_pred

# Top 5 feature importances for the model from the training set
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

def print_model_results(model_name, accuracy, cv_accuracy, report_df):
    print(f"\n{model_name} cross-validated accuracy: {cv_accuracy:.4f}")
    print(f"{model_name} accuracy: {accuracy:.4f}")
    print("Classification report table:")
    print(report_df)

def print_confusion_matrix(model_name, y_test, y_pred):
    print(f"{model_name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def show_model_report_results(model_name, accuracy, cv_accuracy, report, save_path=None):
    report_df = convert_model_report_to_df(report).round(4)
    print_model_results(model_name, accuracy, cv_accuracy, report_df)
    if not save_path:
        save_path = os.path.join('results', f'{model_name.lower()}_classification_report.png')

    create_matplotlib_table(report_df, save_path)

# Show the feature importances in a bar chart, a table and save them
def show_feature_importance_results(model_name, model, X_train):
    print_top_5_feature_importances(model_name, model, X_train)
    top_5_features = top_5_feature_importances(model, X_train)[1]
    top_5_importances_df = convert_top_5_feature_importances_to_df(top_5_features).round(4)
    if_save_path = os.path.join('results', f'{model_name.lower()}_top_5_features.png')
    create_matplotlib_table(top_5_importances_df, if_save_path)
    if_bar_chart_save_path = os.path.join('results', f'{model_name.lower()}_top_5_features_bar_chart.png')
    plot_feature_importances(model_name, model, X_train, if_bar_chart_save_path)

# Show the confusion matrix in a table and a plot and save the plot
def show_confusion_matrix_results(model_name, y_test, y_pred, save_path=None):
    print_confusion_matrix(model_name, y_test, y_pred)
    if not save_path:
        save_path = os.path.join('results', f'{model_name.lower()}_confusion_matrix.png')
    plot_confusion_matrix(model_name, y_test, y_pred, save_path)

# Trains, evaluates, and compares models using given data, cross-validation, and data type, returning the best model and its accuracy.
def evaluate_models_on_data(models, X_train, X_test, y_train, y_test, cv, data_type='Original'):
    best_model = None
    best_model_name = None
    best_accuracy = 0
    report_save_path = None
    confusion_save_path = None

    for model_name, model in models.items():
        print(f"\n{model_name} model ({data_type} data):")
        accuracy, cv_accuracy, report, y_pred = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, cv)
        if data_type != "Original":
            report_save_path = os.path.join('results', f'{model_name.lower()}_{data_type.lower()}_classification_report.png')
            confusion_save_path = os.path.join('results', f'{model_name.lower()}_{data_type.lower()}_confusion_matrix.png')
        show_model_report_results(model_name, accuracy, cv_accuracy, report, save_path=report_save_path)
        show_confusion_matrix_results(model_name, y_test, y_pred, save_path=confusion_save_path)

        if accuracy > best_accuracy and data_type == 'Original':
            best_model = model
            best_model_name = model_name
            best_accuracy = accuracy

    return best_model, best_model_name, best_accuracy

# Splits input data, oversamples if needed, and evaluates various classifiers on the data, reporting performance and feature importances.
def train_and_evaluate_models(data):
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_resampled, _, y_train_resampled, _ = split_data(data, oversample=True)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss')
    }

    # Using StratifiedKFold to maintain class distribution across folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model, best_model_name, _ = evaluate_models_on_data(models, X_train, X_test, y_train, y_test, cv, data_type='Original')
    evaluate_models_on_data(models, X_train_resampled, X_test, y_train_resampled, y_test, cv, data_type='Oversampled')

    print(f"\n===================== Best model: {best_model_name} =====================")

    show_feature_importance_results(best_model_name, best_model, X_train)


def main():
    output_data = utils.pandas_load_csv("output_data.csv")
    
    train_and_evaluate_models(output_data)

if __name__ == '__main__':
    main()