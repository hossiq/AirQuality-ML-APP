import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Combined_Final.csv')

# Creating AQI classes 
bins = [-1, 50, 100, float('inf')]  # The bins for the classes
labels = [0, 1, 2]  # The labels for the classes (0: 0-50, 1: 51-100, 2: >100)
df['AQI_Class'] = pd.cut(df['AQI_O3'], bins=bins, labels=labels)

# Defining features and target variable
X = df[['AWND', 'PRCP', 'SNOW', 'TMAX', 'TMIN']]
y = df['AQI_Class']

# Spliting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
# Using CART model
cart = DecisionTreeClassifier(random_state=42)
cart.fit(X_train, y_train)

# Getting feature importances
importances_rfc = rfc.feature_importances_
importances_cart = cart.feature_importances_


importances_rfc_percentage = 100 * (importances_rfc / importances_rfc.sum())
importances_cart_percentage = 100 * (importances_cart / importances_cart.sum())

# Creating a DataFrame for a comparative view of feature importance for these models
df_importances = pd.DataFrame({
    'Feature': X.columns,
    'RF Importance (%)': importances_rfc_percentage,
    'CART Importance (%)': importances_cart_percentage,
})
df_importances_sorted = df_importances.sort_values(by='RF Importance (%)', ascending=False).reset_index(drop=True)

print(df_importances_sorted.to_string())