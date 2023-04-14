import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import utils
from sklearn.impute import SimpleImputer

# Load data from a CSV file
def load_data(filename):
    return pd.read_csv(utils.get_data_file_path(filename))

# Drop rows with missing veracity values
def drop_na_veracity(data):
    return data.dropna(subset=['veracity'])

# Convert clubs_mentioned column to a list of clubs
def convert_clubs_mentioned(data):
    data = data.copy()
    data.loc[:, 'clubs_mentioned'] = data['clubs_mentioned'].apply(
        lambda x: eval(x) if x.startswith('[') else x.split(', '))
    return data

# Clean market_value column by converting to numerical values
def clean_market_value(value):
    if pd.isna(value) or value == '-':
        return np.nan

    value = value.replace('€', '')
    if 'm' in value:
        value = float(value.replace('m', '')) * 1_000_000
    elif 'k' in value:
        value = float(value.replace('k', '')) * 1_000
    return value

# Process market_value column
def process_market_value(data):
    data = data.copy()
    data[data.columns[data.columns.get_loc('market_value')]] = data['market_value'].apply(clean_market_value)
    return data

# One-hot encode categorical columns
def encode_columns(data, columns_to_encode):
    encoded_data = []
    for col in columns_to_encode:
        if col == 'clubs_mentioned':
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(data[col])
            encoded_df = pd.DataFrame(encoded, columns=mlb.classes_)
        else:
            ohe = OneHotEncoder(sparse_output=False)
            encoded = ohe.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([col]))
        
        encoded_data.append(encoded_df)
    return encoded_data

# Concatenate original data with encoded columns
def concat_encoded_data(data, encoded_data):
    data.reset_index(drop=True, inplace=True)
    data_encoded = pd.concat([data] + encoded_data, axis=1)
    return data_encoded

# Preprocess data: drop missing values, convert clubs_mentioned, process market_value, and one-hot encode categorical columns
def preprocess_data(data):
    data = drop_na_veracity(data)
    data = convert_clubs_mentioned(data)
    data = process_market_value(data)
    columns_to_encode = ['clubs_mentioned', 'nationality', 'position', 'source']
    encoded_data = encode_columns(data, columns_to_encode)
    data_encoded = concat_encoded_data(data, encoded_data)
    return data_encoded

# Drop unnecessary columns
def drop_columns(data_encoded):
    return data_encoded.drop(['date', 'id', 'clubs_mentioned', 'nationality', 'position', 'source', 'veracity'], axis=1)

# Get X (features) and y (target)
def get_X_y(data, data_encoded):
    data = data.dropna(subset=['veracity'])
    X = drop_columns(data_encoded)
    y = data['veracity'].astype(int)
    return X, y

# Impute missing values using mean imputation
def impute_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    return X_imputed

def split_data(data, data_encoded):
    X, y = get_X_y(data, data_encoded)
    X_imputed = impute_missing_values(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def print_model_results(model_name, accuracy, report):
    print(f"{model_name} accuracy: {accuracy}")
    print(report)

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
        report = classification_report(y_test, y_pred)

        print_model_results(model_name, accuracy, report)
        print_top_5_feature_importances(model_name, model, X_train)

def print_top_5_feature_importances(model_name, model, X_train):
    print(f"\n{model_name} top 5 features:")
    feature_importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importances[:5]:
        print(f"{feature}: {importance}")

if __name__ == '__main__':
    data = load_data("final_data.csv")
    data_encoded = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data, data_encoded)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
