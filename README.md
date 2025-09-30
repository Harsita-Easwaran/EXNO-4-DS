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
df=pd.read_csv('income(1) (1).csv')
df
````
<img width="2533" height="1174" alt="image" src="https://github.com/user-attachments/assets/10b60ad5-5cc4-47f7-84a7-35be08d8017a" />


```
from sklearn.preprocessing import LabelEncoder

df_encoder = df.copy()
le = LabelEncoder()

for col in df_encoder.select_dtypes(include="object").columns:
  df_encoder[col] = le.fit_transform(df_encoder[col])

x = df_encoder.drop("SalStat", axis=1)
y = df_encoder["SalStat"]
x
```
<img width="2135" height="834" alt="image" src="https://github.com/user-attachments/assets/0e3e8a4c-5df2-47f7-844b-5ba91390b39e" />


```
from sklearn.feature_selection import SelectKBest, chi2

chi2_selector = SelectKBest(chi2, k=5)
chi2_selector.fit(x,y)

selected_features_chi2 = x.columns[chi2_selector.get_support()]
print("Selected features (Chi-Square):", list(selected_features_chi2))

mi_scores = pd.Series(chi2_selector.scores_, index=x.columns)
print(mi_scores.sort_values(ascending=False))
```
<img width="1592" height="507" alt="image" src="https://github.com/user-attachments/assets/39f7466f-781a-4ab8-b043-1a3ba554e539" />




```
from sklearn.feature_selection import SelectKBest, f_classif

anova_selector = SelectKBest(f_classif, k=5)
anova_selector.fit(x,y)

selected_features_anova = x.columns[anova_selector.get_support()]
print("Selected features (ANOVA F-test):", list(selected_features_anova))

mi_scores = pd.Series(anova_selector.scores_, index=x.columns)
print(mi_scores.sort_values(ascending=False))
```
<img width="1550" height="509" alt="image" src="https://github.com/user-attachments/assets/7341f828-a430-4a8f-b1a2-2f833e91a7fb" />





```
from sklearn.feature_selection import mutual_info_classif
mi_selector = SelectKBest(mutual_info_classif, k=5)
mi_selector.fit(x,y)

selected_feature_mi = x.columns[mi_selector.get_support()]
print("Selected features (Mutual Info):", list(selected_feature_mi))

mi_scores = pd.Series(mi_selector.scores_, index=x.columns)
print("\nMutual Information Scores:\n",mi_scores.sort_values(ascending=False))
```
<img width="1697" height="585" alt="image" src="https://github.com/user-attachments/assets/d13e1f4c-e1af-4c6a-9ec5-b72602aa6bb0" />





```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(max_iter=100)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(x,y)

selected_features_rfe = x.columns[rfe.support_]
print("Selected features (RFE):", list(selected_features_rfe))
```
<img width="1404" height="217" alt="image" src="https://github.com/user-attachments/assets/1b7a9353-a3bf-4127-a945-b1b2a43a3796" />






```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x,y)

importances = pd.Series(rf.feature_importances_, index=x.columns)
selected_features_rf = importances.sort_values(ascending=False).head(5).index
print("Top 5 features (Random Forest Importance):", list(selected_features_rf))
```
<img width="1785" height="68" alt="image" src="https://github.com/user-attachments/assets/588b0e45-3078-4bfd-b39b-45d96a94aad5" />







```
from sklearn.linear_model import LassoCV
import numpy as np

lasso = LassoCV(cv=5).fit(x,y)
importance = np.abs(lasso.coef_)

selected_features_lasso = x.columns[importance > 0]
print("Selected features (Lasso):", list(selected_features_lasso))
```
<img width="1293" height="60" alt="image" src="https://github.com/user-attachments/assets/0586ffdc-1bbc-4812-830b-10a8855d6367" />






```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

model = LogisticRegression(max_iter=1000)

rfe = SequentialFeatureSelector(model, n_features_to_select=5)
rfe.fit(x_scaled, y)

selected_features_sfs = x.columns[rfe.get_support()]
print("Selected features (SFS):", list(selected_features_sfs))
```
<img width="1605" height="61" alt="image" src="https://github.com/user-attachments/assets/df67c0f9-53af-403c-9c6e-bcc8665a54af" />






```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

df=pd.read_csv("income(1) (1).csv")

le = LabelEncoder()
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("SalStat", axis=1)
y = df_encoded["SalStat"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X_train, y_train)

```
<img width="610" height="162" alt="image" src="https://github.com/user-attachments/assets/706cc04a-a5f5-467d-b6e9-71b46f6b5183" />





```
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
<img width="902" height="581" alt="image" src="https://github.com/user-attachments/assets/ce799780-0f74-4448-8352-ab42678417eb" />

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
