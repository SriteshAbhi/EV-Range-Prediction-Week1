## Week-1 Project  
### Electric Vehicle Range Prediction  

This week i worked on a dataset about Electric Vehicles.  
The data has details like battery size, top speed, torque, range, seats, etc.  
My main goal was to clean the data, understand it and then build a simple model to predict the electric car range (in km).

---

### Steps i did  

1. Loaded the dataset using pandas  
2. Checked for missing values and filled them  
3. Did some visualizations using seaborn and matplotlib  
4. Observed how battery capacity and top speed affect the range  
5. Built a Linear Regression model using sklearn  
6. Tested the model and compared actual vs predicted results  

---

### Model Results  

- **Mean Squared Error:** 1248.79  
- **R² Score:** 0.88  
- The predicted range values are almost close to the real values  

---

### Files in my folder  

Week-1/
│
--- electric_vehicle_data.csv -> dataset file
--- ev_price_prediction.py -> model training and testing file
--- step2_clean_visualize.py -> cleaning and visualization file
--- EDA_visualizations.png -> histogram of EV range
--- Range_Prediction_Comparison.png -> scatter plot of actual vs predicted
--- README.md -> my notes file

---

### Libraries i used  

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  

---

### What i learnt  

From this task i learnt how to  
- clean and prepare data  
- visualize important features  
- train a basic ML model and check its accuracy  

It was really interesting to see how battery and power data can be used to predict vehicle range.  
Had fun doing this project 😄  

---
