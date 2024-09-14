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
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![image](https://github.com/user-attachments/assets/d532a478-5ae4-4836-911f-491c67a361cb)
![image](https://github.com/user-attachments/assets/e0dc47f1-8fa6-4334-b687-1307721dfb39)
![image](https://github.com/user-attachments/assets/a6d166bc-283d-426d-921c-a4069d161e10)
![image](https://github.com/user-attachments/assets/09eab529-1eec-43ad-be86-4aa602954d3c)
![image](https://github.com/user-attachments/assets/f1d1202d-fe60-4e80-991e-7f233bf4c85f)
![image](https://github.com/user-attachments/assets/856479a9-8897-4ac2-8866-5a55ff34d746)
![image](https://github.com/user-attachments/assets/f269a19f-b238-4a51-b8df-b3e134f91e33)
![image](https://github.com/user-attachments/assets/36cf88a2-e134-434f-a3d0-9b53f061957a)
![image](https://github.com/user-attachments/assets/961f9165-f4e7-476a-85c9-006665c467f5)

![image](https://github.com/user-attachments/assets/83bf1c21-f8a8-4a43-83e3-1e020c1bf0cd)
![image](https://github.com/user-attachments/assets/c9921979-60e2-4d81-a508-a20f93fd1f3c)
![image](https://github.com/user-attachments/assets/0474b5a7-d732-494b-9a2a-e725e0399c4d)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
