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


#part 2
# handling null values
print(df.describe())
print(df.isnull().sum())
print(df.dropna(axis=1, inplace=False))
#normalization

scaler = MinMaxScaler()
df['AvgMonthSpend'] = scaler.fit_transform(df[['AvgMonthSpend']].values.reshape(-1,1))
df['YearlyIncome'] = scaler.fit_transform(df[['YearlyIncome']].values.reshape(-1,1))
print(df)
print(len(df['YearlyIncome'].value_counts().index))

bin_YearlyIncome = np.arange(min(df['YearlyIncome'],max(df['YearlyIncome']),20000))
len(bin_YearlyIncome)
YearlyIncome_label=['below low','low','Medium','above medium','High']
df['YearlyBinned'] = pd.cut(df['YearlyIncome'],bin_YearlyIncome,labels=YearlyIncome_label)

bin_AvgMonthSpend = np.arange(min(df['AvgMonthSpend'],max(df['AvgMonthSpend']),20000))
len(bin_AvgMonthSpend)
AvgMonthSpend = ['Low','Medium','High']
df['AvgMonthSpendBinned '] = pd.cut(df['AvgMonthSpend'],bin_AvgMonthSpend,labels=AvgMonthSpend)

onehotencoder=OneHotEncoder()
onehotencoder.fit_transform(df['HomeOwnerFlag'].values.reshape(-1,1))
onehotencoder.fit_transform(df['Gender'].values.reshape(-1,1))


def jaccard(s1,s2):
    return len(s1.intersection(s2)/len(s1.union(s2)))
df['Jaccard'] = df.apply(lambda x: jaccard(set(x['HomeOwnerFlag']),set(x['Gender'])),axis=1)

