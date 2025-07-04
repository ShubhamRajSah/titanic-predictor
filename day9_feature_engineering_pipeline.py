import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# 📥 Load Dataset
df = pd.read_csv('C:/Users/LENOVO/OneDrive/Documents/shubham-personal/OneDrive/Desktop/git_projects/day2_titanic_analysis/train.csv')

# 🛠 Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch']
df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare']=df['Fare'].fillna(df["Fare"].median())


# New engineered features
df['Age*Pclass'] = df['Age'] * df['Pclass']
df['Fare_Per_Person'] = df['Fare'] / df['FamilySize'].replace(0, 1)
df['Is_Mother'] = ((df['Sex'] == 'female') & (df['Age'] > 18) & (df['Parch'] > 0)).astype(int)

# 🔡 Encode Categorical Variables
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# 📦 Prepare Data
feature_cols = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'FamilySize', 'IsAlone',
    'Age*Pclass', 'Fare_Per_Person', 'Is_Mother'
]

X = df[feature_cols]
y = df['Survived']

# 🔧 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🧪 Evaluation
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 📊 Feature Importance Plot
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance with Engineered Features")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()