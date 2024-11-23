# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the `pandas` library to handle the dataset and load the dataset `Placement_Data.csv` using `pd.read_csv()`.

2. Display the first few rows of the dataset using the `head()` function for an initial preview.

3. Create a copy of the original dataset using the `copy()` method to preserve the original data.

4. Drop unnecessary columns such as `sl_no` and `salary` using the `drop()` method to focus on relevant features.

5. Check for missing values in the dataset using the `isnull().sum()` method to identify any data cleaning requirements.

6. Check for duplicate records in the dataset using the `duplicated().sum()` method to ensure the data is unique.

7. Import `LabelEncoder` from `sklearn.preprocessing` to handle categorical data.

8. Apply the `LabelEncoder` to encode categorical columns such as `gender`, `ssc_b`, `hsc_b`, `hsc_s`, `degree_t`, `workex`, `specialisation`, and `status` into numerical values.

9. Display the processed dataset after encoding to confirm the transformation.

10. Separate the features (`X`) and the target variable (`y`) from the dataset.  
    - Extract all columns except the target column as features (`X`).  
    - Select the target column `status` as the dependent variable (`y`).

11. Import `train_test_split` from `sklearn.model_selection` to split the dataset into training and testing sets.

12. Split the dataset into training and testing subsets by specifying the test size (e.g., 20%) and a random state for reproducibility.

13. Import `LogisticRegression` from `sklearn.linear_model` to build the logistic regression model.

14. Initialize the logistic regression model with the `liblinear` solver.

15. Train the logistic regression model using the training features (`x_train`) and the target variable (`y_train`) with the `fit()` method.

16. Predict the target variable on the testing data (`x_test`) using the `predict()` method.

17. Calculate the accuracy of the model using `accuracy_score` from `sklearn.metrics` by comparing the true labels (`y_test`) and the predicted labels (`y_pred`).

18. Compute the confusion matrix using `confusion_matrix` from `sklearn.metrics` to evaluate the performance of the classification model.

19. Generate a classification report using `classification_report` from `sklearn.metrics` to view precision, recall, F1-score, and support for each class.

20. Print the classification report to analyze the model's performance.

21. Make a prediction for a new data point by passing the feature values as a list to the `predict()` method of the trained model.

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
