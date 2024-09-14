# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

## Program:
```Python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HASNA MUBARAK AZEEM
RegisterNumber: 212223240052

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
*/
```

## Output:
![image](https://github.com/user-attachments/assets/d532a478-5ae4-4836-911f-491c67a361cb)
![image](https://github.com/user-attachments/assets/e0dc47f1-8fa6-4334-b687-1307721dfb39)
![image](https://github.com/user-attachments/assets/a6d166bc-283d-426d-921c-a4069d161e10)
![image](https://github.com/user-attachments/assets/09eab529-1eec-43ad-be86-4aa602954d3c)
![image](https://github.com/user-attachments/assets/f1d1202d-fe60-4e80-991e-7f233bf4c85f)
![image](https://github.com/user-attachments/assets/856479a9-8897-4ac2-8866-5a55ff34d746)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
