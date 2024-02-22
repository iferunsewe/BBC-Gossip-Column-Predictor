import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import os

FIGSIZE = (12, 14)
CATEGORICAL_RESULTS_TO_SHOW = 10
FONTSIZE = 24

# Preprocess data for visualization by dropping NaNs and converting y_col to numeric
def preprocess_for_visulization(data, x_col, y_col):
    # Drop rows with NaN values in the x_col and y_col columns
    data = data[[x_col, y_col]].dropna()

    # Convert 'veracity' column to numeric
    data[y_col] = pd.to_numeric(data[y_col])

    return data

# Plot a boxplot figure with specified data, columns, title, and save to a file
def plot_boxplot_figure(data, x_col, y_col, title, x_col_label, y_col_label, save_path):
    plt.figure(figsize=(8, 8))
    sns.boxplot(data=data, x=y_col, y=x_col)
    plt.title(title, fontsize=FONTSIZE)
    plt.xlabel(y_col_label, fontsize=FONTSIZE)
    plt.ylabel(x_col_label, fontsize=FONTSIZE) 
    plt.savefig(save_path)
    plt.close()

# Print descriptive statistics for boxplot data
def print_boxplot_statistics(data, x_col, y_col, title):
    boxplot_stats = data.groupby(y_col)[x_col].describe()
    print(f"Boxplot statistics for {title}:\n")
    print(boxplot_stats)

# Convert specified column in data to numeric
def convert_column_to_numeric(data, column):
    data[column] = pd.to_numeric(data[column], errors='coerce')
    return data

# Drop rows with NaN values in specified columns
def drop_na_rows(data, columns):
    data_no_na = data[columns].dropna()
    return data_no_na

 # Calculate Pearson correlation and p-value between x_col and y_col
def calculate_pearsonr_correlation(data, x_col, y_col):
    pearsonr, p_value = scipy.stats.pearsonr(data[y_col], data[x_col])
    return pearsonr, p_value

def calculate_point_biserial_correlation(data, x_col, y_col):
    """
    Calculate and print the point-biserial correlation coefficient and p-value
    between a continuous variable and the dichotomous 'veracity' variable.
    """
    # Calculate point-biserial correlation
    correlation, p_value = scipy.stats.pointbiserialr(data[y_col], data[x_col])
    return correlation, p_value

# Interpret and print correlation and p-value results
def interpret_relationship(corr, p_value, x_col, y_col):
    if corr > 0:
        relationship = "positive"
    elif corr < 0:
        relationship = "negative"
    else:
        relationship = "no"

    if abs(corr) >= 0.7:
        strength = "strong"
    elif abs(corr) >= 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    print(f"The correlation of {corr} indicates {relationship} relationship between {x_col} and {y_col} is {strength}.")

    if p_value < 0.05:
        print("The P-value is less than 0.05, which suggests that there is a statistically significant relationship between the two variables.")
    else:
        print("The P-value is greater than 0.05, which suggests that there is no statistically significant relationship between the two variables.")

# Calculate, interpret, and print Pearson correlation for specified columns
def print_correlation(data, x_col, y_col):
    corr, p_value = calculate_point_biserial_correlation(data, x_col, y_col)
    interpret_relationship(corr, p_value, x_col, y_col)

# Create a summary table of true rumour proportions for specified columns
def create_true_rumour_summary_table(data, x_col, y_col):
    true_rumour_counts = data.groupby(x_col)[y_col].sum()
    total_counts = data[x_col].value_counts()
    proportion_true_rumours = (true_rumour_counts / total_counts)
    filtered_proportions = proportion_true_rumours[(proportion_true_rumours > 0) & (total_counts > CATEGORICAL_RESULTS_TO_SHOW)]
    sorted_proportions = filtered_proportions.sort_values(ascending=False)
    percentage_true_rumours = (filtered_proportions * 100).round(2)
    summary = pd.concat([total_counts, percentage_true_rumours], axis=1)
    summary = summary.loc[sorted_proportions.index]
    summary.columns = ['Count', 'Percentage of true rumours']
    summary = summary.sort_values('Percentage of true rumours', ascending=False)
    print(f"Relationship summary between {x_col} and {y_col}:\n{summary}")

    return summary, sorted_proportions

# Plot a bar chart of sorted proportions with specified title and save to a file
def plot_bar_chart(sorted_proportions, title, x_col_label, y_col_label, save_path):
    plt.figure(figsize=(34, 40))
    ax1 = plt.gca()
    sorted_proportions.plot(kind='bar', ax=ax1)
    ax1.set_title(title, fontsize=30)
    ax1.set_ylabel(y_col_label, fontsize=40)
    ax1.set_xlabel(x_col_label, fontsize=40)
    ax1.set_xticklabels([shorten_label(label.get_text()) for label in ax1.get_xticklabels()], fontsize=FONTSIZE)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=30)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{(x*100).round(2)}%'))
    plt.savefig(save_path)
    plt.close()

# Create a table in matplotlib with specified data and save to a file
def create_matplotlib_table(data, save_path):
    _, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis('off')
    table = ax.table(cellText=data.values, rowLabels=[shorten_label(label) for label in data.index], colLabels=data.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(data.columns))))
    table.scale(1, 1.5)

    plt.savefig(save_path) 
    plt.close()

# Calculate class proportions for a given target variable (y)
def calculate_class_proportions(y):
    class_counts = y.value_counts()
    total_count = len(y)
    class_proportions = class_counts / total_count
    return class_proportions

# Plot a bar chart of class distribution with optional title and save path
def plot_class_distribution(y, title='Class Distribution', save_path=None):
    class_proportions = calculate_class_proportions(y)
    plt.figure(figsize=FIGSIZE)
    plt.bar(class_proportions.index, class_proportions.values)
    plt.xlabel('Class', fontsize=FONTSIZE)
    plt.ylabel('Proportion', fontsize=FONTSIZE)
    plt.xticks([0, 1])
    plt.title(title, fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)

    if save_path:
        plt.savefig(save_path)

def shorten_label(label):
    if len(label) > 15:
        return label[:15] + '...'
    return label

# Plot a bar chart of feature importances with optional title and save path
def plot_confusion_matrix(model_name, y_test, y_pred, save_path):
    print("\nPlotting confusion matrix...")
    plt.figure(figsize=FIGSIZE)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title(f"{model_name} Confusion Matrix", fontsize=FONTSIZE)
    plt.xlabel("Predicted Label", fontsize=FONTSIZE)
    plt.ylabel("True Label", fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.savefig(save_path)
    plt.close()

# Plot a bar chart of feature importances with optional title and save path
def plot_feature_importances(model_name, model, X_train, save_path):
    model_importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    importances_df = pd.DataFrame(model_importances, columns=['Feature', 'Importance'])
    plt.figure(figsize=(30, 8))
    sns.barplot(x='Importance', y='Feature', data=importances_df.head(10))
    plt.title(f"{model_name} Feature Importances", fontsize=FONTSIZE)
    plt.xlabel('Importance', fontsize=FONTSIZE)
    plt.ylabel('Feature', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(save_path)
    plt.close()

# Visualize and analyze continuous relationships between features and target column
def show_continuous_relationship(data, continuous_features, y_col):
    for feature in continuous_features:
        if feature in data.columns:
            data_continuous = preprocess_for_visulization(data, feature, y_col)
            save_path = os.path.join('results', f"{feature}_vs_{y_col}_boxplot.png")
            if feature == "market_value":
                x_col_label = "Market Value (in millions)"
            elif feature == "time_to_transfer_window":
                x_col_label = "Time to Transfer Window (in days)"
            else:
                x_col_label = feature
    
            plot_boxplot_figure(data_continuous, feature, y_col, f"{feature} vs. {y_col}", x_col_label, y_col, save_path)
            print_boxplot_statistics(data_continuous, feature, y_col, f"{feature} vs. {y_col}")
        else:
            print(f"Feature not found in the data: {feature}")
        data = convert_column_to_numeric(data, y_col)
        data_no_na = drop_na_rows(data, [feature, y_col])
        print_correlation(data_no_na, feature, y_col)

# Visualize and analyze categorical relationships between features and target column
def show_categorical_relationship(data, categorical_features, y_col):
    for feature in categorical_features:
        if feature in data.columns:
            data_categorical = data[[feature, y_col]].dropna()
            summary, sorted_proportions = create_true_rumour_summary_table(data_categorical, feature, y_col)

            # Check if the DataFrame is not empty
            if not sorted_proportions.empty:
                save_path_bar_chart = os.path.join('results', f"{feature}_vs_{y_col}_bar_chart.png")
                plot_bar_chart(sorted_proportions, f"{feature} vs. {y_col}", feature, 'Percentage of true rumours', save_path_bar_chart)

                save_path_summary_table = os.path.join('results', f"{feature}_vs_{y_col}_summary_table.png")
                create_matplotlib_table(summary, save_path_summary_table)
            else:
                print(f"Warning: The DataFrame for {feature} and {y_col} is empty. Skipping the plot.")
        else:
            print(f"Feature not found in the data: {feature}")

def visualize_and_analyze(data, continuous_features, categorical_features, y_col):
    # Create 'results' directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    target_col_data = data[y_col]
    
    plot_class_distribution(target_col_data, title='Class Distribution in Training Set', save_path=os.path.join('results', 'class_distribution_train.png'))
    show_continuous_relationship(data, continuous_features, y_col)
    show_categorical_relationship(data, categorical_features, y_col)

def main():
    data = utils.pandas_load_csv("output_data.csv")

    #Plot the relationship between the specified features and veracity
    continuous_features_to_analyze = ['age', 'time_to_transfer_window', 'market_value']
    categorical_features_to_analyze = ['nationality', 'position', 'source']
    visualize_and_analyze(data, continuous_features_to_analyze, categorical_features_to_analyze, 'veracity')
    
if __name__ == '__main__':
    main()
