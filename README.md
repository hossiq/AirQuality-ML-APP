# AirQuality-ML-APP
Exploring the process of leveraging machine learning (ML) to forecast air quality. And step-by-step instructions on integrating this model into a user-friendly application using Flask

**Background:**

Polluted Air is responsible for various health issues such as increases the risk of heart disease, respiratory infections, lung cancer, tuberculosis, chronic respiratory disease, diabetes, kidney disease and low birth weight, all of which can lead to premature death. And it is a major environmental threat, for instance it contaminate the surface of bodies of water and soil.Meteorology such as Temperature, Wind Speed, Relative Humidity, etc. contribute to air pollution, along with emissions from cars, industries, power plants, crop burning, etc. Air Quality Index (AQI) which is a scorecard of six categories that describe ambient air quality relative to the relevant standards. This project is focused on predicting AQI using meteorological variables.

**Data Collection and Processing:**

AQI and meteorogical datadata is collected from three different Counties in Arkansas,USA for 2018-2022 from NOAA(National Oceanic and Atmospheric Administration) and EPA(Environmental Protection Agency). Daily Max AQI data is considered for counties with multiple AQI monitors, and weather data is taken from nearest met station of the specific AQI monitor. There are a number of variables, but for this project Wind Speed(AWND), Precipitation(PRCP), SNOW, Temperature Max (TMAX) and Min(TMIN) data is considered. The basic data quality check is performed and missing values are processed by interpolation using python.

**Data Analysis**

I used python coding to analyze various aspects of the data. For instance, only 0.43% of data has AQI>100, whereas 92.22% data is in 0-50 class. Correlation heatmap shows correlation co-efficient between variables,and we can see Max Temperature (positively)  and  Precipitation (negatively) are strognly correlated to AQI. 

<img src="https://github.com/iqbal-T19/image/blob/main/AQI%20counts_Overall.png?raw=true" alt="Image 1" style="width: 400px; height: 300px; object-fit: cover;" /> <img src="https://github.com/iqbal-T19/image/blob/main/Corr%20plot.png?raw=true" alt="Image 2" style="width: 400px; height: 300px; object-fit: cover;" />

The seasonality plot and timeseries plot shows, AQI has seasonal effect and Summer season shows highest AQI variability!

<img src="https://github.com/iqbal-T19/image/blob/main/TimeSeries_Plot.png?raw=true" alt="TimeSeries_Plot" style="width: 400px; height: 300px; object-fit: cover;" /><img src="https://github.com/iqbal-T19/image/blob/main/Seasonality_Pulaski.png?raw=true" alt="Pulaski_Seasonality" style="width: 400px; height: 300px; object-fit: cover;" />
<img src="https://github.com/iqbal-T19/image/blob/main/Seasonality_Crittenden.png?raw=true" alt="Crittenden_seasnality" style="width: 400px; height: 300px; object-fit: cover;"/><img src="https://github.com/iqbal-T19/image/blob/main/Seasonality_Washington.png?raw=true"  alt="Washington_seasnality" style="width: 400px; height: 300px; object-fit: cover;"/>

**Feature Importance**

This project evaluated feature importance of the variables to select features that are most important to use in the model. Since, This is a classification problem, feature importance is evaluated utilizing RandomForestClassifier, and CART (DecisionTreeClassifier) models.

<pre>
```
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
```
</pre>

<p align="center">
  <img src="https://github.com/iqbal-T19/image/blob/main/Feature_Importance.PNG?raw=true" alt="Feature Importance Plot" />
</p>
The output shows SNOW can be removed from the modeling as it doesn't have any importance for AQI prediction.


**Model Development**

> **CART (Classification and Regression Trees)**
  
This study utilized CART model for the initial trial because its ability to handle non-linear relationships and various types of variables without stringent data prerequisites. For this model,test size kept at 30%. And 'stratify=y' option used in the train_test_split because of imbalanced datasets which ensure that the train and test sets are representative of the entire dataset.
  <pre>
```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Initialize the Decision Tree Classifier
cart_model = DecisionTreeClassifier(random_state=42)
# Train the model
cart_model.fit(X_train, y_train)
# Predict on the test set
cart_pred = cart_model.predict(X_test) 
    ```
</pre>


The model output has accuracy of ~ 87%, but it couldn't predict for AQI > 100 class(minority class).Precision indicates how many were actually positive out of all the instances the model predicted as positive. Recall indicates how many did the model correctly identify out of all the actual positive instances, and F1-score indicates a balance between precision and recall

</pre>
<p align="center">
  <img src="https://github.com/iqbal-T19/image/blob/main/CART_out.PNG?raw=true" alt=" Plot" />
</p>

Then, to check for overfitting a decision tree depths up to 20 is considered and the results shown in the following plot

<p align="center">
  <img src="https://github.com/iqbal-T19/image/blob/main/CART_Overfitting.PNG?raw=true" alt="Plot" width="450"/>
</p>
The plot shows as the depth increases, the model perform better on the test set due to its improved ability to generalize. However, beyond a certain depth (after about depth=3), the test accuracy starts to plateau and then decreases, indicating that the model is starting to overfit the training data and is losing its generalization capability on unseen data.
To address the overfitting issue, a new model is applied.

> **Random Forest (RF) model**

RF is an ensemble learning method (combine numerous classifiers to enhance a model's performance) that fits a number of decision tree classifiers on various sub-samples of the dataset and uses majority voting for classification tasks to improve predictive accuracy and control over-fitting. In this method,  'class_weight' parameter is assigend as 'balanced' to ensure that the model pays more attention to the minority class.
  <pre>
```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))
# Initialize and train the Random Forest Classifier with class weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
model.fit(X_train, y_train)
   ```
</pre>

Using the above RF method with 'class_weight' setup, the model still couldn't predict >100 class. Also, recall and precision is value is low for 51-100 class. Then, SMOTE(Synthetic Minority Over-sampling Technique) and ADASYN(Adaptive Synthetic Sampling) sampling technique is applied to deal with class inbalance issue to get better f1-score for minority classes. SMOTE generate synthetic samples by finding  k nearest neighbors (in the minority class) and synthetic instances are created by choosing one of these neighbors at random and then drawing a line in the feature space between the sample and its neighbor, and synthetic samples are generated along this line. ADASYN method also creates synthetic samples, but it calculates the density distribution of the minority class instances and generates more synthetic data for those instances that are surrounded by neighbors from the majority class. Meaning it focuses on the regions where the classifier has more difficulty to recognize minority class.  
  <pre>
```
               #SMOTE Technique   
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Apply SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_smote, y_train_smote)
   ```
</pre>

 <pre>
```
               #ADASYN Technique   
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Apply ADASYN for oversampling the minority class
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
# Initialize and train the Random Forest Classifier 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_adasyn, y_train_adasyn)
# Predict on test set
y_pred = model.predict(X_test)
   ```
</pre>

<img src="https://github.com/iqbal-T19/image/blob/main/RandomForest.PNG?raw=true" alt="rf" style="width: 280px; object-fit: cover;"/><img src="https://github.com/iqbal-T19/image/blob/main/RandomForest_SMOTE.PNG?raw=true"  alt="smote" style="width: 280px; object-fit: cover;"/><img src="https://github.com/iqbal-T19/image/blob/main/RandomForest_ADASYN.PNG?raw=true"  alt="ADASYN" style="width: 2800px; object-fit: cover;"/>

As none of these methods are best way to capture minority class in test dataset, this study considered 'season' as an independent feature (for demonstration purpose) and re-run the models for each of the sampling scenarios. ADASYN method with class_weight provides better result.

 <pre>
```

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
  ```
</pre>

<p align="center">
  <img src="https://github.com/iqbal-T19/image/blob/main/Random Forest Classification with ADASYN and class_weight.PNG?raw=true" alt="Plot" width="450"/>
</p>

The F1-score is 0.93 which indicating a strong balance between precision and recall for 0-50 class while F1-score is 0.27 and 0.18 indicates poor performance for 51-100 and >100 classes, respectivly.


**Model Deployment/APP Development**


>**Pickle for Model**


      Pickle is a Python module used to convert Python objects into a byte stream known as pickling which is used to save the trained model to a file. This process is called the model serialization.

 <pre>
```
import pickle
#save model 
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
  ```
</pre>


>**Building a Web Application with Flask**



  Python Flask is used to create a web application that serves as an interface to the machine learning model.
  
 <pre>
```
from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to one-hot encode the season input
def encode_season(season):
    return [1 if season == s else 0 for s in ['Spring', 'Summer', 'Fall', 'Winter']]

# Route for handling the landing page logic
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    color = 'black'  # Default text color

    if request.method == 'POST':
        # Get input values
        input_values = request.form.to_dict()

        # Handle season encoding
        season = input_values.pop('season')
        season_encoded = encode_season(season)

        # Convert other inputs to float and combine with season encoding
        input_values = [float(value) for value in input_values.values()]
        input_values = season_encoded + input_values
        input_values = np.array(input_values).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_values)[0]

        # Color coding based on AQI_O3 category
        if prediction == '0-50':
            color = 'green'
        elif prediction == '51-100':
            color = 'yellow'
        else:  # prediction == '>100'
            color = 'red'

    return render_template('index.html', prediction=prediction, color=color)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
 ```
</pre>

>**HTML Front-end**

 The front-end of web application is developed using HTML and CSS that provides a user interface for inputting data into the model and displaying the output.


```html
    #html and css script
   
<!DOCTYPE html>
<html>
<head>
    <title>Predict AQI_O3</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            color: #333;
            line-height: 1.6;
        }
        .container {
            width: 60%;
            margin: auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        .input-group {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            width: 48%;
        }
        .center {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        h1 {
            text-align: center;
            color: #4CAF50; /* Green color */
        }
        .date {
            text-align: center;
            font-size: 16px;
            margin-bottom: 10px;
            color: #555;
        }
        label {
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="number"], select {
            width: 100%;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
            margin-bottom: 10px; /* Added for spacing */
        }
        input[type="submit"] {
            background-color: #4CAF50; /* Green color */
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        .prediction {
            text-align: center;
            font-size: 18px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Air Quality Prediction App</h1>
    <p id="currentDate" class="date"></p>
    <div class="container">
        <form method="POST">
            <h2>Enter Input Values</h2>
            <!-- First row of input fields -->
            <div class="row">
                <div class="input-group">
                    <label for="AWND">Average Wind Speed (m/s):</label>
                    <input type="number" name="AWND" placeholder="AWND">
                </div>
                <div class="input-group">
                    <label for="PRCP">Precipitation (mm):</label>
                    <input type="number" name="PRCP" placeholder="PRCP">
                </div>
            </div>

            <!-- Second row of input fields -->
            <div class="row">
                <div class="input-group">
                    <label for="TMAX">Max Temperature (°F):</label>
                    <input type="number" name="TMAX" placeholder="TMAX">
                </div>
                <div class="input-group">
                    <label for="TMIN">Min Temperature (°F):</label>
                    <input type="number" name="TMIN" placeholder="TMIN">
                </div>
            </div>

            <!-- Third row for season dropdown -->
            <div class="row">
                <div class="input-group">
                    <label for="season">Season:</label>
                    <select name="season">
                        <option value="Spring">Spring</option>
                        <option value="Summer">Summer</option>
                        <option value="Fall">Fall</option>
                        <option value="Winter">Winter</option>
                    </select>
                </div>
            </div>

            <!-- Predict button -->
            <div class="center">
                <input type="submit" value="Press to Predict AQI">
            </div>

            <!-- Display prediction -->
            {% if prediction is not none %}
                <div class="prediction" style="color: {{ color }};">Prediction: {{ prediction }}</div>
            {% endif %}
        </form>
    </div>

    <script>
        // JavaScript to display the current date
        document.getElementById("currentDate").innerHTML = "Today's Date: " + new Date().toLocaleDateString();
    </script>
</body>
</html>
```

<p align="center">
  <img src="https://github.com/iqbal-T19/image/blob/main/AirQualityAPP.PNG?raw=true" alt="App"/>
</p>

**Future Work**

This study demonstrated step by step process of building an APP to predict air quality. The above results show the model result is not predicting minority classes well. This limitation underscores the need for a more robust dataset and a diversified set of predictive features. For further study, considering more data (10-15 years) to get better sample of minority class and more independent variables which are related to air quality such as humidity, cloud cover, etc. may improve the model.



