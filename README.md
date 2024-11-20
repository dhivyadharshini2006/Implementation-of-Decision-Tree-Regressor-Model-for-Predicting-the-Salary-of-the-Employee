# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries: `pandas` for data manipulation and modules from `sklearn` for machine learning tasks.

2. Load the dataset `Salary.csv` using `pd.read_csv()` and display its structure using `.head()` and `.info()` to examine the dataset's contents and data types.

3. Check for missing values in the dataset using `.isnull().sum()` to ensure data completeness.

4. Encode the categorical column `Position` into numerical values using `LabelEncoder` from `sklearn.preprocessing`. Use `.fit_transform()` to perform the encoding.

5. Verify the updated dataset structure and preview the changes using `.head()`.

6. Separate the features (`x`) and target variable (`y`). Select the columns `Position` and `Level` for `x`, and the column `Salary` for `y`.

7. Split the dataset into training and testing subsets using `train_test_split()` with `test_size=0.2` to allocate 20% of the data for testing and `random_state=100` for reproducibility.

8. Import the `DecisionTreeRegressor` from `sklearn.tree` and initialize the model.

9. Train the `DecisionTreeRegressor` on the training data (`x_train`, `y_train`) using the `fit()` method.

10. Predict the target values for the test data (`x_test`) using the trained model’s `predict()` method. Store the predictions in `y_pred`.

11. Import metrics from `sklearn` to evaluate the model’s performance. Calculate the mean squared error (MSE) using `metrics.mean_squared_error(y_test, y_pred)`.

12. Compute the coefficient of determination (\(R^2\) score) to assess the goodness of fit using `metrics.r2_score(y_test, y_pred)`.

13. Test the model with new input data by providing a feature array (e.g., `[[5, 6]]`) to the model’s `predict()` method to predict the salary for that input. 

14. Display the results, including MSE, \(R^2\) score, and predictions for the test data and new input.
## Program:
```Python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Dhivya Dharshini B
RegisterNumber: 212223240031
*/

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv('Salary.csv')
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![image](https://github.com/user-attachments/assets/5d43147a-0ef1-4f78-980a-04445613c668)

![image](https://github.com/user-attachments/assets/35630bb8-1523-46b0-8497-e1809aaaca8b)

![image](https://github.com/user-attachments/assets/6381af5a-33ba-48f2-b730-2b001d1ff726)

![image](https://github.com/user-attachments/assets/0d28cdbf-4aed-4d6e-a4c4-7787cd2b0b98)

![image](https://github.com/user-attachments/assets/326a2ce5-de47-48ec-8ef4-455613b98686)

![image](https://github.com/user-attachments/assets/3f97c3ca-d44c-4717-b1d2-f66afc1c5c37)

![image](https://github.com/user-attachments/assets/f7c8fb8f-f83a-4354-9776-f14ad5e299f7)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
