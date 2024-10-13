# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

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
