#import data manupulation libraries
import pandas as pd
import numpy as np

#import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#import filter warning libraries
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


#Data import 
url='https://raw.githubusercontent.com/Saimehtre18/Heart_Failure_Prediction/refs/heads/main/heart.csv'
df=pd.read_csv(url)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['ChestPainType']=le.fit_transform(df['ChestPainType'])
df['RestingECG']=le.fit_transform(df['RestingECG'])
df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])
df['ST_Slope']=le.fit_transform(df['ST_Slope'])
df['Sex']=le.fit_transform(df['Sex'])


# Capping Outliers 
def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
    return df

outlier_cols = ['Sex', 'RestingBP', 'Cholesterol', 'FastingBS','RestingECG','MaxHR','Oldpeak']
for col in outlier_cols:
    df = cap_outliers(df, col)
    
X=df.drop(columns=['HeartDisease'])
y=df['HeartDisease']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier()
RF.fit(X_train,y_train)
y_pred=RF.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

accuracy_score_LR=accuracy_score(y_test,y_pred)
confusion_matrix_LR=confusion_matrix(y_test,y_pred)

from sklearn.model_selection import KFold, cross_val_score

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(RF, X, y, cv=kf, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Average CV Accuracy:", np.mean(scores))