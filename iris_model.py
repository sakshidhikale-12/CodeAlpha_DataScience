
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Iris.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Count:")
print(df['Species'].value_counts())


X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

sns.pairplot(df, hue='Species')
plt.show()
