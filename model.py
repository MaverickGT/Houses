from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('house.csv')

print("Data Frame shape: \n", df.shape, "\n")
print("Data Frame columns: \n", df.columns, "\n")
print("Data Frame types: \n", df.dtypes, "\n")
print("Data Frame Head: \n", df.head(), "\n")
print("Data Frame Description: \n", df.describe(), "\n")


#convert all values to numeric
df['Brick'] = df['Brick'].map({'Yes': 1, 'No': 0})
df['Neighborhood'] = df['Neighborhood'].map({'North': 1, 'South': 2, 'East': 3, 'West': 4})


# Handle missing values
df.fillna(0, inplace=True)  # Replace missing values with 0

#check for outliers and handle them
print(f'Number of rows before removing outliers: {len(df)}')
for col in df.columns:
  if df[col].dtype != 'object':
    z = np.abs(stats.zscore(df[col]))
    threshold = 3
    outliers = df[z > threshold]
    print(f'Number of outliers in {col}: {len(outliers)}')
    df = df.drop(outliers.index)
print(f'Number of rows after removing outliers: {len(df)}')

#check for null values and handle them
print(f"\nNumber of null values before handling: {df.isnull().values.sum()}")
if df.isnull().values.any():
  df = df.fillna(df.median())
print(f"\nNumber of null values after handling: {df.isnull().values.sum()}")

#check for duplicate values and handle them
print(f"\nNumber of duplicate values before handling: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"\nNumber of duplicate values after handling: {df.duplicated().sum()}")


# Bivariate relationships
for col1 in df.columns:
  for col2 in df.columns:
    if col1 != col2 and df[col1].dtype in [np.float64, np.int64] and df[col2].dtype in [np.float64, np.int64]:
      sns.scatterplot(data=df, x=col1, y=col2)
      plt.title(f'{col1} vs {col2}')
      #plt.show()

# Feature correlations
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Choose a regression model to predict house prices based on the dataset
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
