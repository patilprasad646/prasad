import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
data1=pd.read_csv(r"C:\Users\IndianDataScience\Desktop\DataScience_Project\BostonHousing_regression\train.csv")
data2=data1.drop('ID',axis=1)
data=data2.drop(['crim','zn','indus','chas','nox','age','dis','rad','tax','ptratio','black'],axis=1)
X=data.drop('medv',axis=1)
Y=data['medv']
corr=data.corr()
sns.heatmap(corr,annot=True)
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression
limodel=LinearRegression()
limodel.fit(X_train,Y_train)
pred=limodel.predict(X_test)
print(mean_squared_error(Y_test,pred))
print(r2_score(Y_test,pred))

