# ML Algorithm 

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df=pd.read_excel("C:/Users/HP1/Desktop/diabetes.xlsx")
print(df.head())
X = df.drop('Glucose', axis=1)
y = df['Glucose']

sns.boxplot(x=df['Glucose'])
plt.title('Box Plot for exploring outliers')
plt.xlabel(df['Glucose'])
plt.show()

Q1 = df['Glucose'].quantile(0.25)
Q3 = df['Glucose'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Glucose'] < lower_bound) | (df['Glucose'] > upper_bound)]
print("outliers")
print(outliers)

corr_matrix=df.corr(method='pearson')
print("Correlation:", corr_matrix)

sns.heatmap(corr_matrix,annot=True)
plt.title("Exam Relations")
plt.show()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi2_selector = SelectKBest(score_func=chi2, k=5)
X_kbest = chi2_selector.fit_transform(X_scaled, y)

selected_features = X.columns[chi2_selector.get_support()]
print("Best features of using chi-square:")
print(selected_features)

X = df.drop("Glucose", axis=1)
y = df["Glucose"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)

rfe = RFE(estimator=model, n_features_to_select=5)

rfe.fit(X_scaled, y)

selected_features = X.columns[rfe.support_]
print("Best features of using RFE:")
print(selected_features)

X = df.drop('Glucose', axis=1)
y = df['Glucose']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Glucose'] = y


plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', palette='Set1')
plt.title('Two-dimensional representation using PCA')
plt.xlabel('(PC1)')
plt.ylabel('(PC2)')
plt.grid(True)
plt.show()


X = df.drop("Glucose", axis=1)
y = df["Glucose"]
features = X.columns 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)

model_lasso = SelectFromModel(lasso, prefit=True)
selected_lasso = features[model_lasso.get_support()]

print(" \nFeatures selected by Lasso:")
print(list(selected_lasso))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

model_rf = SelectFromModel(rf, prefit=True)
selected_rf = features[model_rf.get_support()]

print("\n Features selected by Random Forest:")
print(list(selected_rf)) 

df['BMI_Age'] = df['BMI'] * df['Age']
df['lowBMIWithDiabetes'] = df.apply(lambda row: 1 if row['BMI'] < 18 and row['Outcome'] == 1 else 0, axis=1)

df['HighGlucose'] = df['Glucose'].apply(lambda x: 1 if x > 130 else 0)

df['YoungWithDiabetes'] = df.apply(lambda row: 1 if row['Age'] < 30 and row['Outcome'] == 1 else 0, axis=1)

print(df[['BMI_Age', 'HighGlucose', 'YoungWithDiabetes','lowBMIWithDiabetes']])

count_1 = (df['YoungWithDiabetes'] == 1).sum()
print('the total of 1 is: ',count_1)
