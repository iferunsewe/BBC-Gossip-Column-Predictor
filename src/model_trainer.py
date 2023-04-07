import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Load the final.csv dataset
data = pd.read_csv("../data/final.csv")

# Drop rows with NaN values in the 'veracity' column
data = data.dropna(subset=['veracity'])

# Convert the 'clubs_mentioned' column from string to a list of clubs
data['clubs_mentioned'] = data['clubs_mentioned'].apply(lambda x: eval(x))

# One-hot encode the 'clubs_mentioned' column using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
clubs_encoded = mlb.fit_transform(data['clubs_mentioned'])
clubs_encoded_df = pd.DataFrame(clubs_encoded, columns=mlb.classes_)

# Reset the index of the 'data' dataframe
data.reset_index(drop=True, inplace=True)

# Concatenate the original data with the one-hot encoded clubs_mentioned dataframe
data_encoded = pd.concat([data, clubs_encoded_df], axis=1)

# Drop the 'news_id', 'player_name', and 'clubs_mentioned' columns
X = data_encoded.drop(['news_id', 'player_name', 'clubs_mentioned', 'veracity'], axis=1)

# Use the 'veracity' column as the target variable
y = data['veracity']

# Convert boolean values to integers
y = y.astype(int)

# Split the dataset into training and testing sets (80-20 split)
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
X_train = np.array(X_train)
y_train = np.array(y_train)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Train and evaluate AdaBoost Classifier
ab = AdaBoostClassifier(random_state=42)
ab.fit(X_train, y_train)
y_pred_ab = ab.predict(X_test)
print("AdaBoost accuracy:", accuracy_score(y_test, y_pred_ab))
print(classification_report(y_test, y_pred_ab))

# Train and evaluate XGBoost Classifier
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# Perform hyperparameter optimization for XGBoost using GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

xgb_opt = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
grid_search = GridSearchCV(xgb_opt, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters for XGBoost:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_xgb = grid_search.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)
print("Best XGBoost accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print(classification_report(y_test, y_pred_best_xgb))
