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


The model output has accuracy of ~ 87%, but it couldn't predict for AQI > 100 class(minority class).
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
# Initialize and train the Random Forest Classifier with class weights
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_adasyn, y_train_adasyn)
# Predict on test set
y_pred = model.predict(X_test)
   ```
</pre>

<img src="https://github.com/iqbal-T19/image/blob/main/RandomForest.PNG?raw=true" alt="rf" style="width: 400px; height: 300px; object-fit: cover;"/><img src="https://github.com/iqbal-T19/image/blob/main/RandomForest_SMOTE.PNG?raw=true"  alt="smote" style="width: 400px; height: 300px; object-fit: cover;"/><img src="https://github.com/iqbal-T19/image/blob/main/RandomForest_ADASYN.PNG?raw=true"  alt="ADASYN" style="width: 400px; height: 300px; object-fit: cover;"/>






