import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.utils.class_weight import compute_class_weight
import pickle

# Function to categorize AQI_O3
def categorize_aqi(aqi):
    if aqi <= 50:
        return '0-50'
    elif 51 <= aqi <= 100:
        return '51-100'
    else:
        return '>100'

# Function to determine the season based on the month
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'  # Assuming months 12, 1, 2 are Winter

# Load the dataset
file_path = 'Combined_Final.csv'  
data = pd.read_csv(file_path)

# Convert 'Date' to datetime to extract the month
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month

# Create the 'Season' column and drop unnecessary columns
data['Season'] = data['Month'].apply(get_season)
data.drop(['Date', 'Month', 'SNOW', 'COUNTY'], axis=1, inplace=True)

# Apply one-hot encoding to 'Season'
data = pd.get_dummies(data, columns=['Season'])

# Convert AQI_O3 to categories
data['AQI_O3'] = data['AQI_O3'].apply(categorize_aqi)

# Separate features and target variable
X = data.drop('AQI_O3', axis=1)
y = data['AQI_O3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply ADASYN for oversampling the minority class
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# Compute class weights for cost-sensitive learning
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_adasyn), y=y_train_adasyn)
class_weights_dict = dict(zip(np.unique(y_train_adasyn), class_weights))

# Initialize and train the Random Forest Classifier with class weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
model.fit(X_train_adasyn, y_train_adasyn)

#save model 
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:RandomForest(with ADASYN & class_weight)\n", report)