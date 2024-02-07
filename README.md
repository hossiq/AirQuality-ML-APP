# AirQuality-ML-APP
Exploring the process of leveraging machine learning (ML) to forecast air quality. And step-by-step instructions on integrating this model into a user-friendly application using Flask

**Background:**

Polluted Air is responsible for various health issues such as increases the risk of heart disease, respiratory infections, lung cancer, tuberculosis, chronic respiratory disease, diabetes, kidney disease and low birth weight, all of which can lead to premature death. And it is a major environmental threat, for instance it contaminate the surface of bodies of water and soil.Meteorology such as Temperature, Wind Speed, Relative Humidity, etc. contribute to air pollution, along with emissions from cars, industries, power plants, crop burning, etc. Air Quality Index (AQI) which is a scorecard of six categories that describe ambient air quality relative to the relevant standards. This project is focused on predicting AQI using meteorological variables.

**Data Collection and Processing:**

AQI and meteorogical datadata is collected from three different Counties in Arkansas,USA for 2018-2022 from NOAA(National Oceanic and Atmospheric Administration) and EPA(Environmental Protection Agency). Daily Max AQI data is considered for counties with multiple AQI monitors, and weather data is taken from nearest met station of the specific AQI monitor. There are a number of variables, but for this project Wind Speed(AWND), Precipitation(PRCP), SNOW, Temperature Max (TMAX) and Min(TMIN) data is considered. The basic data quality check is performed and missing values are processed by interpolation using python.

**Data Analysis**
I used python coding to analyze various aspects of the data. For instance, only 0.43% of data has AQI>100, whereas 92.22% data is in 0-50 class. Correlation heatmap shows correlation co-efficient between variables,and we can see Max Temperature (positively)  and  Precipitation (negatively) are strognly correlated to AQI. 
<img src="https://github.com/iqbal-T19/image/blob/main/AQI%20counts_Overall.png?raw=true" width="50%" alt="AQI_Data_Count_Not_Working" /><img src="https://github.com/iqbal-T19/image/blob/main/Corr%20plot.png?raw=true" width="50%" alt="Coorelation_Plot"/>

The seasonality plot and timeseries plot shows, AQI has seasonal effect and Summer season shows highest AQI variability!
<img src="https://github.com/iqbal-T19/image/blob/main/TimeSeries_Plot.png?raw=true" width="500" alt="TimeSeries_Plot">
<img src="https://github.com/iqbal-T19/image/blob/main/Seasonality_Pulaski.png?raw=true" width="500" alt="Pulaski_Seasonality"/><img src="https://github.com/iqbal-T19/image/blob/main/Seasonality_Crittenden.png?raw=true" width="500" alt="Crittenden_seasnality"/>




