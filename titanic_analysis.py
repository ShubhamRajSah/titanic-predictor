import pandas as pd
from colorama import Fore, Style

df = pd.read_csv('C:/Users/LENOVO/OneDrive/Documents/shubham-personal/OneDrive/Desktop/git_projects/day2_titanic_analysis/titanic_sample.csv')
df['Name']=df['Name'].astype(str) + df['Unnamed: 4'].astype(str)
df['Name']=df['Name'].str.replace(r'^(\w+)\s+',r'\1,',regex=True)
df.drop(columns=['Unnamed: 4'],inplace=True)
# print(df.head()) # this gives a messy view of table i.e not formtted nicely.
print(Fore.YELLOW+df.to_string(index=False)+Style.RESET_ALL) # it gives a nice view of table also in a formatted way.

# print("\nData shape:", df.shape)
# print("\nAverage Age:", df['Age'].mean())

survival_counts = df['Survived'].value_counts()
print(survival_counts)
# print("keys available:",survival_counts.index.tolist())
# print(Fore.CYAN + "🧾 Survival Breakdown:" + Style.RESET_ALL)
# print(Fore.GREEN + f"Survived: {survival_counts.get(1,0)}" + Style.RESET_ALL)
# print(Fore.RED + f"Did Not Survive: {survival_counts.get(0,0)}\n" + Style.RESET_ALL)


# print("\n Survival Rate by Class:")
# print(df.groupby('Pclass')['Survived'].mean())

# # survival by gender
# print("\n survival by gender")
# print(df.groupby('Sex')['Survived'].mean())

# # gender + class breakdown
# print("survival by gender and class")
# print(df.groupby(['Sex','Pclass'])['Survived'].mean())

# # survival among children
# children=df[df['Age']<18]
# print('total children: ',children.shape[0])
# print("children survial rate: ",children['Survived'].mean())

# Bar Graph of survival by Fare
import matplotlib.pyplot as plt

# print(df.groupby('Fare')['Survived'].mean())

# df.boxplot(column='Survived',by='Fare')
# plt.title("Survival distribution by Fare")
# plt.suptitle('')
# plt.xlabel('Survived')
# plt.ylabel("Fare")
# plt.show()



# Pie Chart of survival by Sex

# survival_by_sex=df.groupby('Sex')['Survived'].mean()
# print(survival_by_sex)
# survival_by_sex.plot(kind='pie', autopct='%1.1f%%', startangle=90)
# plt.title("survival rate by gender")
# plt.ylabel('')
# plt.show()

print("missing values from each column: ")
print(df.isnull().sum()) #checking for null values in the models

print(df['Name'].unique()) #prints unique names in the Name column

print("current column names: ")
print(df.columns.tolist())

df['Title'] = df['Name'].str.extract(r',\s*(\w+)\.?\s', expand=False)
print(df[['Name', 'Title']].head())
print("\nUnique Titles:", df['Title'].unique()) #prints unique titles
print(df.groupby('Title')['Survived'].mean().sort_values(ascending=False))


# ⚠ Fabricated Data for Learning Only:
# The SibSp and Parch values below are randomly generated and do not reflect actual Titanic passenger data.
import numpy as np

np.random.seed(42)  # For reproducibility
df['SibSp'] = np.random.randint(0, 3, size=len(df))
df['Parch'] = np.random.randint(0, 3, size=len(df))

# df['FamilySize']=df['SibSp']+df['Parch'] + 1
# print(df.groupby('FamilySize')['Survived'].mean().sort_index())


# Calculate survival rate per family size
# family_survival = df.groupby('FamilySize')['Survived'].mean()

# Plot it
# family_survival.plot(kind='bar', color='teal', edgecolor='black')
# plt.title('Survival Rate by Family Size')
# plt.xlabel('Family Size')
# plt.ylabel('Survival Rate')
# plt.xticks(rotation=0)
# plt.ylim(0, 1)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

# df['isAlone']=(df['FamilySize']==1).astype(int)
# print(df.groupby('isAlone')['Survived'].mean())

# df.groupby('isAlone')['Survived'].mean().plot(kind='bar',color='pink ',edgecolor='black')

# plt.title('Survival Rate : Alone vs Not Alone')
# plt.ylabel('Survival Rate')
# plt.ylim(0,1)
# plt.xticks([0,1],['Not Alone','Alone'],rotation=0)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()


# convert 'sex' to numbers

df['Sex']=df['Sex'].map({'male':0 ,'female':1})

# simplify rare titles first
df['Title']=df['Title'].replace(['Lady','Countess','Capt','Col'],'Rare')

df['Title']=df['Title'].replace(['Mile', 'Ms'],'Miss')

# converitng titles to numbers
title_mapping={'Mr':1,'Miss':2,'Mrs':3}
df['Title']=df['Title'].map(title_mapping)

def expand_dataframe(df, times=20):
    expanded = pd.concat([df.copy() for i in range(times)], ignore_index=True)
    
    # Add small noise ti Age and Fare 
    np.random.seed(42)
    expanded['Age'] += np.random.normal(0,5, size=len(expanded))
    expanded['Fare'] += np.random.normal(0,10, size=len(expanded))
    
    # clip values to keep things realistic
    expanded['Age']= expanded['Age'].clip(0,80)
    expanded['Fare']= expanded['Fare'].clip(0,80)

    return expanded

# calling the function and updating Dataframe
df=expand_dataframe(df, times=20)

print(Fore.GREEN+df.to_string(index=False)+Style.RESET_ALL)

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
df[['Age','Fare']]=scaler.fit_transform(df[['Age','Fare']])

from sklearn.model_selection import train_test_split

features=['Pclass','Sex','Age','Fare','Title']
x=df[features]
y=df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# Predict on test data
y_pred = model.predict(x_test)

# Evaluate
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 confusion matrix:\n", confusion_matrix(y_test,y_pred))


