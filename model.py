import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('house.csv')

print("Data Frame shape: \n", df.shape, "\n")
print("Data Frame columns: \n", df.columns, "\n")
print("Data Frame types: \n", df.dtypes, "\n")
print("Data Frame Head: \n", df.head(), "\n")
print("Data Frame Description: \n", df.describe(), "\n")

# Handle missing values
numeric_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Handle outliers
for col in df.columns:
  if df[col].dtype != 'object':
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
# Check for null values
if df.isnull().values.any():
  df = df.fillna(df.median())

# Check for duplicate rows
if df.duplicated().any():
  df = df.drop_duplicates()

# Check for invalid/outlier values
for col in df.columns:
  if df[col].dtype in [np.float64, np.int64]:
    df = df[(df[col] > df[col].quantile(0.01))
            & (df[col] < df[col].quantile(0.99))]

print(df.info())
print(df.describe())

# Univariate distribution
for col in df.columns:
  if df[col].dtype in [np.float64, np.int64]:
    sns.displot(df[col])
    plt.title(f'Distribution of {col}')
    plt.show()
    plt.close()

# Bivariate relationships
for col1 in df.columns:
  for col2 in df.columns:
    if col1 != col2 and df[col1].dtype in [np.float64, np.int64] and df[col2].dtype in [np.float64, np.int64]:
      sns.scatterplot(data=df, x=col1, y=col2)
      plt.title(f'{col1} vs {col2}')
      plt.show()
      plt.close()

# Feature correlations
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
plt.close()






