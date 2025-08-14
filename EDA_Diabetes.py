
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

#open file
df=pd.read_excel("C:/Users/HP1/Desktop/diabetes.xlsx")

#show EDA
print(df.head())
print(df.describe())
print(df.isnull().sum())

