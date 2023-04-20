import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import utils
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

# Load data from a CSV file
def load_data(filename):
    return pd.read_csv(utils.get_data_file_path(filename))

# Drop rows with missing veracity values
def drop_na_veracity(data):
    return data.dropna(subset=['veracity'])

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

# Preprocess data: drop missing values, convert clubs_mentioned and one-hot encode categorical columns
def preprocess_data(data):
    data = drop_na_veracity(data)
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

        print_model_results(model_name, accuracy, report, y_test, y_pred)
        print_top_5_feature_importances(model_name, model, X_train)

def print_top_5_feature_importances(model_name, model, X_train):
    total_features = len(X_train.columns)
    print(f"\n{model_name} top 5 features out of {total_features} features:")
    feature_importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importances[:5]:
        print(f"{feature}: {importance}")

    # Plot feature importances
    model_importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    importances_df = pd.DataFrame(model_importances, columns=['Feature', 'Importance'])
    plt.figure()
    sns.barplot(x='Importance', y='Feature', data=importances_df.head(10))
    plt.title(f"{model_name} Feature Importances")
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()


def preprocess_for_visulization(data, x_col, y_col):
    # Drop rows with NaN values in the x_col and y_col columns
    data = data[[x_col, y_col]].dropna()

    # Convert 'veracity' column to numeric
    data[y_col] = pd.to_numeric(data[y_col])

    return data

def print_model_results(model_name, accuracy, report, y_test, y_pred):
    print(f"{model_name} accuracy: {accuracy}")
    # Remove support column and Macro average and Weighted average rows
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=['support'])
    report_df = report_df.drop(['macro avg', 'weighted avg'])
    
    # Round the classification report results
    report_df = report_df.round(decimals=2)
    
    print("Classification report table:")
    print(report_df)

    # Display the classification report as a table
    plot_summary_table(report_df)
    
    print(f"{model_name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_boxplot(data, x_col, y_col, title):
    plt.figure()
    sns.boxplot(data=data, x=y_col, y=x_col)
    plt.title(title)
    plt.xlabel(y_col)
    plt.ylabel(x_col)
    plt.show()
    
    # Print boxplot statistics
    boxplot_stats = data.groupby(y_col)[x_col].describe()
    print(f"Boxplot statistics for {title}:\n")
    print(boxplot_stats)

def convert_column_to_numeric(data, column):
    data[column] = pd.to_numeric(data[column], errors='coerce')
    return data

def drop_na_rows(data, columns):
    data_no_na = data[columns].dropna()
    return data_no_na

def calculate_point_biserial_correlation(data, x_col, y_col):
    point_biserial_corr, p_value = scipy.stats.pointbiserialr(data[y_col], data[x_col])
    return point_biserial_corr, p_value

def interpret_relationship(point_biserial_corr, p_value, x_col, y_col):
    if point_biserial_corr > 0:
        relationship = "positive"
    elif point_biserial_corr < 0:
        relationship = "negative"
    else:
        relationship = "no"

    if abs(point_biserial_corr) >= 0.7:
        strength = "strong"
    elif abs(point_biserial_corr) >= 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    print(f"The point-biserial correlation of {point_biserial_corr} indicates {relationship} relationship between {x_col} and {y_col} is {strength}.")

    if p_value < 0.05:
        print("The P-value is less than 0.05, which suggests that there is a statistically significant relationship between the two variables.")
    else:
        print("The P-value is greater than 0.05, which suggests that there is no statistically significant relationship between the two variables.")

def plot_continuous_relationship(data, x_col, y_col, title):
    plot_boxplot(data, x_col, y_col, title)
    data = convert_column_to_numeric(data, y_col)
    data_no_na = drop_na_rows(data, [x_col, y_col])
    point_biserial_corr, p_value = calculate_point_biserial_correlation(data_no_na, x_col, y_col)
    interpret_relationship(point_biserial_corr, p_value, x_col, y_col)

def calculate_summary_table(data, x_col, y_col):
    true_rumour_counts = data.groupby(x_col)[y_col].sum()
    total_counts = data[x_col].value_counts()
    proportion_true_rumours = (true_rumour_counts / total_counts)
    filtered_proportions = proportion_true_rumours[(proportion_true_rumours > 0) & (total_counts > 10)]
    sorted_proportions = filtered_proportions.sort_values(ascending=False)

    summary = pd.concat([total_counts, filtered_proportions], axis=1)
    summary = summary.loc[sorted_proportions.index]
    summary.columns = ['Count', 'Proportion of true rumours']
    summary = summary.sort_values('Proportion of true rumours', ascending=False)
    print(f"Relationship summary between {x_col} and {y_col}:\n{summary}")


    return summary, sorted_proportions


def plot_bar_chart(sorted_proportions, title):
    plt.figure()
    ax1 = plt.gca()
    sorted_proportions.plot(kind='bar', ax=ax1)
    ax1.set_title(title)
    ax1.set_ylabel('Proportion of true rumours')
    ax1.set_xticklabels([shorten_label(label.get_text()) for label in ax1.get_xticklabels()])
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    plt.show()


def plot_summary_table(summary):
    num_rows = len(summary)
    table_height = max(min(num_rows * 0.3, 8), 4)

    _, ax2 = plt.subplots(figsize=(12, table_height))
    ax2.axis('off')
    table = ax2.table(cellText=summary.values, rowLabels=[shorten_label(label) for label in summary.index], colLabels=summary.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.auto_set_column_width(col=list(range(len(summary.columns))))
    table.scale(1, 1.5)

    plt.show()

def plot_categorical_relationship(data, x_col, y_col, title):
    summary, sorted_proportions = calculate_summary_table(data, x_col, y_col)
    plot_bar_chart(sorted_proportions, title)
    plot_summary_table(summary)


def shorten_label(label):
    if len(label) > 15:
        return label[:15] + '...'
    return label

def plot_relationships(data, continuous_features, categorical_features, y_col):
    for feature in continuous_features:
        if feature in data.columns:
            data_continuous = preprocess_for_visulization(data, feature, y_col)
            plot_continuous_relationship(data_continuous, feature, y_col, f"{feature} vs. {y_col}")
        else:
            print(f"Feature not found in the data: {feature}")
    
    for feature in categorical_features:
        if feature in data.columns:
            data_categorical = data[[feature, y_col]].dropna()
            plot_categorical_relationship(data_categorical, feature, y_col, f"{feature} vs. {y_col}")
        else:
            print(f"Feature not found in the data: {feature}")


if __name__ == '__main__':
    data = load_data("output_data.csv")
    data_encoded = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data, data_encoded)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

    #Plot the relationship between the specified features and veracity
    continuous_features_to_analyze = ['age', 'time_to_transfer_window', 'market_value']
    categorical_features_to_analyze = ['nationality', 'position', 'source']
    plot_relationships(data, continuous_features_to_analyze, categorical_features_to_analyze, 'veracity')
