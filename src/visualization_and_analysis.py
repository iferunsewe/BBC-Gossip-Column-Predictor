import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

# Load data from a CSV file
def load_data(filename):
    return pd.read_csv(utils.get_data_file_path(filename))

def preprocess_for_visulization(data, x_col, y_col):
    # Drop rows with NaN values in the x_col and y_col columns
    data = data[[x_col, y_col]].dropna()

    # Convert 'veracity' column to numeric
    data[y_col] = pd.to_numeric(data[y_col])

    return data

def plot_boxplot_figure(data, x_col, y_col, title):
    plt.figure()
    sns.boxplot(data=data, x=y_col, y=x_col)
    plt.title(title)
    plt.xlabel(y_col)
    plt.ylabel(x_col)
    plt.show()

def print_boxplot_statistics(data, x_col, y_col, title):
    boxplot_stats = data.groupby(y_col)[x_col].describe()
    print(f"Boxplot statistics for {title}:\n")
    print(boxplot_stats)

def plot_boxplot(data, x_col, y_col, title):
    plot_boxplot_figure(data, x_col, y_col, title)
    print_boxplot_statistics(data, x_col, y_col, title)

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

def print_correlation(data, x_col, y_col):
    corr = data[x_col].corr(data[y_col])
    print(f"The correlation between {x_col} and {y_col} is {corr}.")

def plot_continuous_relationship(data, x_col, y_col, title):
    plot_boxplot(data, x_col, y_col, title)
    data = convert_column_to_numeric(data, y_col)
    data_no_na = drop_na_rows(data, [x_col, y_col])
    print_correlation(data_no_na, x_col, y_col)

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
    
    # Check if the DataFrame is not empty
    if not sorted_proportions.empty:
        plot_bar_chart(sorted_proportions, title)
        plot_summary_table(summary)
    else:
        print(f"Warning: The DataFrame for {x_col} and {y_col} is empty. Skipping the plot.")

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

def plot_confusion_matrix(model_name, y_test, y_pred):
    print("\nPlotting confusion matrix...")
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_feature_importances(model_name, model, X_train):
    model_importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    importances_df = pd.DataFrame(model_importances, columns=['Feature', 'Importance'])
    plt.figure()
    sns.barplot(x='Importance', y='Feature', data=importances_df.head(10))
    plt.title(f"{model_name} Feature Importances")
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def main():
    data = load_data("output_data.csv")

    #Plot the relationship between the specified features and veracity
    continuous_features_to_analyze = ['age', 'time_to_transfer_window', 'market_value']
    categorical_features_to_analyze = ['nationality', 'position', 'source']
    plot_relationships(data, continuous_features_to_analyze, categorical_features_to_analyze, 'veracity')
    
if __name__ == '__main__':
    main()
