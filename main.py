import numpy as np
import pandas as pd


#part 1
df_Cus = pd.read_csv('AWCustomers.csv')
df_Sales = pd.read_csv('AWSales.csv')
df_Cus.info()
df_Cus.head()
df_Sales.head()
df_Sales.info()
print(df_Cus.columns)
print(df_Sales.columns)
print(df_Sales.drop(['CustomerID'],axis=1,inplace=True))
df = pd.concat([df_Cus,df_Sales],axis=1)
print(df)
print(df.shape)
print(df.drop(['Title','Suffix','Education','MaritalStatus','Occupation','Gender','PhoneNumber','MiddleName'],axis=1,inplace=True))
print(df.head(10))

print(df['AddressLine2'].isnull().sum())
df.drop(['AddressLine2'],axis=1,inplace=True)
print(df.describe())
df.info()

categorical_columns = df.dtypes[df.dtypes == 'object'].index
print(categorical_columns)
numerical_columns = df.dtypes[df.dtypes == 'int64'].index
print(numerical_columns)
discrete_columns = df.dtypes[df.dtypes == 'int64'].index
print(discrete_columns)
continous_columns = df.dtypes[df.dtypes == 'float64'].index
print(continous_columns)