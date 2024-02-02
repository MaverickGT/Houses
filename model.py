from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

df = pd.read_csv('house.csv')

print("Data Frame shape: \n", df.shape, "\n")
print("Data Frame columns: \n", df.columns, "\n")
print("Data Frame types: \n", df.dtypes, "\n")
print("Data Frame Head: \n", df.head(), "\n")
print("Data Frame Description: \n", df.describe(), "\n")


# Convert all values to numeric
df['Brick'] = df['Brick'].map({'Yes': 1, 'No': 0})
df['Neighborhood'] = df['Neighborhood'].map({'North': 1, 'South': 2, 'East': 3, 'West': 4})


# Handle missing values
df.fillna(0, inplace=True)  # Replace missing values with 0

# Check for outliers and handle them
print(f'Number of rows before removing outliers: {len(df)}')
for col in df.columns:
  if df[col].dtype != 'object':
    z = np.abs(stats.zscore(df[col]))
    threshold = 3
    outliers = df[z > threshold]
    print(f'Number of outliers in {col}: {len(outliers)}')
    df = df.drop(outliers.index)
print(f'Number of rows after removing outliers: {len(df)}')

# Check for null values and handle them
print(f"\nNumber of null values before handling: {df.isnull().values.sum()}")
if df.isnull().values.any():
  df = df.fillna(df.median())
print(f"\nNumber of null values after handling: {df.isnull().values.sum()}")

# Check for duplicate values and handle them
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=44)

# # Selecting the best model
# models = [('Linear Regresion', LinearRegression()),
#           ('Gradient Boosting', GradientBoostingRegressor()),
#           ('Random Forest', RandomForestRegressor())]


# for model in models: #for loop through the three models
#     reg = model[1]  #initialize the model object
#     reg.fit(X_train,y_train)  #fitting the training data
#     pred = reg.predict(X_test)  #predict target
#     print(model[0])
#     print('R2: ',r2_score(y_test, pred))  #check r2 score
#     print('RMSE: ', np.sqrt(mean_squared_error(y_test, pred)))  #check root mean squared error
#     print('-'*30)


model = GradientBoostingRegressor()
model.fit(X_train, y_train)

print(model.get_params())

y_pred = model.predict(X_test)

# Calculate metrics
accuracy = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Gradient Boosting Regression")


# Start an MLflow experiment
with mlflow.start_run():
    
    # Log the hyperparameters
    mlflow.log_params(model.get_params())

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic Gradient Boosting Regression model for house.csv data")

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
      sk_model=model,
      artifact_path="house_model",
      signature=signature,
      input_example=X_train,
      registered_model_name="tracking-gbr-model",
    )
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

column_names = list(df.columns)

result = pd.DataFrame(X_test, columns=column_names)
result["actual"] = y_test
result["predicted"] = predictions

result[:4]