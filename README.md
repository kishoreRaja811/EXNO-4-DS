# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("/content/bmi.csv")
 df.head()
```
<img width="632" height="428" alt="image" src="https://github.com/user-attachments/assets/d1e05185-8cfe-4b43-9f64-5c6a5074d59f" />
```
 df_null_sum=df.isnull().sum()
 df_null_sum
```
<img width="656" height="332" alt="image" src="https://github.com/user-attachments/assets/ad1b0f29-dd8c-4807-8cdc-506fec490fa7" />
```
df.dropna()
```
<img width="674" height="581" alt="image" src="https://github.com/user-attachments/assets/fdbd50c2-c55e-43b0-98dd-eeea3b84fb27" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
 max_vals
```
<img width="779" height="267" alt="image" src="https://github.com/user-attachments/assets/6f67d54b-21d4-40a0-b38c-cd2ccef41a91" />

```
from sklearn.preprocessing import StandardScaler
 df1=pd.read_csv("/content/bmi.csv")
 df1.head()
```
<img width="738" height="405" alt="image" src="https://github.com/user-attachments/assets/ea4f3eac-3c5a-458b-a884-245f172fd75a" />

```
sc=StandardScaler()
 df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
 df1.head(10)
```
<img width="766" height="610" alt="image" src="https://github.com/user-attachments/assets/310d4503-71aa-4431-b459-b7b71ea9d99c" />
```
 from sklearn.preprocessing import MinMaxScaler
 scaler=MinMaxScaler()
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df.head(10)
```
<img width="806" height="616" alt="image" src="https://github.com/user-attachments/assets/0edfd044-ebf1-460a-83dc-16828ac4e565" />
```
 from sklearn.preprocessing import MaxAbsScaler
 scaler = MaxAbsScaler()
 df3=pd.read_csv("/content/bmi.csv")
 df3.head()
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df
```
<img width="827" height="752" alt="image" src="https://github.com/user-attachments/assets/b4d2aebb-f448-46e2-9f50-a8a743053832" />

```
 from sklearn.preprocessing import RobustScaler
 scaler = RobustScaler()
 df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
 df3.head()
```
<img width="798" height="467" alt="image" src="https://github.com/user-attachments/assets/7079f9b8-0e29-4d5a-9fa3-e030d2dbfc90" />
```
 df=pd.read_csv("/content/income(1) (1).csv")
 df.info()
```
<img width="774" height="530" alt="image" src="https://github.com/user-attachments/assets/8e039333-b35b-45ef-bfe9-dbd2531b64cd" />
```
 df_null_sum=df.isnull().sum()
 df_null_sum
```
<img width="760" height="685" alt="image" src="https://github.com/user-attachments/assets/09f11e68-a610-4b2b-8556-22f52a4174f8" />
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
<img width="827" height="737" alt="image" src="https://github.com/user-attachments/assets/710db84f-ae01-401f-ace9-9ba5d0632635" />
```
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
<img width="823" height="651" alt="image" src="https://github.com/user-attachments/assets/bd39ef36-5e95-450d-88a7-4532ed26d0e1" />
```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 from sklearn.ensemble import RandomForestClassifier
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
 ```
 <img width="830" height="348" alt="image" src="https://github.com/user-attachments/assets/544eb0cd-982f-43c7-a16c-9eecfdecbc2f" />
```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="814" height="554" alt="image" src="https://github.com/user-attachments/assets/dbcd4790-c9a3-42ff-93c4-502e31028b4c" />
```
import pandas as pd
 from sklearn.feature_selection import SelectKBest, chi2, f_classif
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
<img width="818" height="758" alt="image" src="https://github.com/user-attachments/assets/5d507bb6-167d-447a-8096-cbf12161abd3" />
```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
<img width="813" height="626" alt="image" src="https://github.com/user-attachments/assets/7655d4f8-26ef-41f8-921e-6e63b932e1c3" />
```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_chi2 = 6
 selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
 X_chi2 = selector_chi2.fit_transform(X, y)
 selected_features_chi2 = X.columns[selector_chi2.get_support()]
 print("Selected features using chi-square test:")
 print(selected_features_chi2)

```
<img width="759" height="363" alt="image" src="https://github.com/user-attachments/assets/068c4516-db2a-433c-aada-f0cac1d1a790" />
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="821" height="420" alt="image" src="https://github.com/user-attachments/assets/70eef6b5-9fd2-468b-9f39-386e21c8717c" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="807" height="194" alt="image" src="https://github.com/user-attachments/assets/367aee82-e683-45d3-8500-1d9d2c29ec8b" />
```
!pip install skfeature-chappers
```
<img width="824" height="439" alt="image" src="https://github.com/user-attachments/assets/4f7feca0-5f55-4c0e-bbdd-73942c3a27be" />
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
<img width="827" height="754" alt="image" src="https://github.com/user-attachments/assets/19dff5d1-70f0-4804-834b-344701421f33" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
<img width="822" height="322" alt="image" src="https://github.com/user-attachments/assets/87a15a92-d0db-484e-9189-acb25f5bb3ab" />

```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="825" height="736" alt="image" src="https://github.com/user-attachments/assets/38acddfd-495b-4327-8c02-8088f0e4dac2" />
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
<img width="1479" height="895" alt="image" src="https://github.com/user-attachments/assets/8490bc82-ddb0-4a87-9852-310bfd2e7cf8" />

# RESULT:
     Code successfully excited.
