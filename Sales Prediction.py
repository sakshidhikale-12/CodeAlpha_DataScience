
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("advertising.csv")

print("fIRST 5 rOWS:")
print(df.head())

print("\ndATA iNFO:")
print(df.info())

print("\nSTATISTICS:")
print(df.describe())

print("\nmISSING VALUES:")
print(df.isnull().sum())


plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation heatmap")
plt.show()


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nmODEL pERFORMANCE:")
print("Error:", mse)
print("R2 Score:", r2)

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

print("\naDVERTISING iMPACT:")
print("TV:", model.coef_[0])
print("Radio:", model.coef_[1])
print("Newspaper:", model.coef_[2])

sample_data = np.array([[200, 30, 20]])
predicted_sales = model.predict(sample_data)

print("\nsAMPLE pREDICTION:")
print("For TV=200, Radio=30, Newspaper=20")
print("Predicted Sales:", predicted_sales[0])
