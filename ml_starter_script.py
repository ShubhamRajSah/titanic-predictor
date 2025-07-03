
# 🧠 STEP 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('C:/Users/LENOVO/OneDrive/Documents/shubham-personal/OneDrive/Desktop/git_projects/day2_titanic_analysis/train.csv')

# ⚙ STEP 2: Feature Engineering Function
np.random.seed(42)  # For reproducibility
# df['SibSp'] = np.random.randint(0, 3, size=len(df))
# df['Parch'] = np.random.randint(0, 3, size=len(df))

def engineer_features(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # SibSp and Parch means siblingspouse and parcentchild respectivley.
    df['isAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Don', 'Dona', 'Dr'], 'Rare')
    return df

df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna('S')
df['Deck']=df['Cabin'].str[0]
df['Deck']=df['Deck'].fillna('M') # M for missing
df['TicketPrefix']=df['Ticket'].str.extract(r'([A-Za-z\.\/]+)',expand=False)
df['TicketPrefix']=df['TicketPrefix'].fillna('X') # X for missing

# 📦 STEP 3: Preprocessing Function
def preprocess(df):
    df = engineer_features(df)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked']=LabelEncoder().fit_transform(df['Embarked'])
    df['Title'] = LabelEncoder().fit_transform(df['Title'])
    df['TicketPrefix'] = LabelEncoder().fit_transform(df['TicketPrefix'])
    df['Deck'] = LabelEncoder().fit_transform(df['Deck'])

    df[['Age', 'Fare']] = StandardScaler().fit_transform(df[['Age', 'Fare']])
    return df

clean_df=preprocess(df)

# step 4:T= Training and Evaluating the model
features =['Pclass','Sex','Age','Fare','Title','FamilySize','isAlone','Embarked','TicketPrefix','Deck']
x=clean_df[features]
y=clean_df['Survived']
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42) # this can be used inside the function below but is kept above so that other calculations can also be done 
def train_and_evaluate_model(model, x_train, x_test ,y_test, y_train):
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    print(' Accuracy: ', accuracy_score(y_test,y_pred))  
    print('\n classification report:\n', classification_report(y_test,y_pred))
    print('\n confusion matrix:\n',confusion_matrix(y_test,y_pred))


try:
    
    # features =['Pclass','Sex','Age','Fare','Title','FamilySize','isAlone','Embarked','TicketPrefix','Deck']
    # x=clean_df[features]
    # y=clean_df['Survived']
    model=RandomForestClassifier(n_estimators=100, random_state=42)
    print(f'the model used for test and train is {model}')
    train_and_evaluate_model(model, x_train, x_test ,y_test, y_train)
except Exception as e:
    print(e)


import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances
# importances = model.feature_importances_
# feature_names = x.columns

# # Create a DataFrame for easy plotting
# feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feat_df = feat_df.sort_values(by='Importance', ascending=False)

# # Plot
# plt.figure(figsize=(10,6))
# sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
# plt.title('Feature Importances from Random Forest')
# plt.tight_layout()
# plt.show() 


# from sklearn.metrics import precision_recall_curve, average_precision_score

# # Get prediction probabilities for the positive class (class 1)
# y_scores = model.predict_proba(x_test)[:, 1]

# # Calculate precision and recall
# precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
# avg_precision = average_precision_score(y_test, y_scores)

# # Plot
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, color='purple', linewidth=2)
# plt.title(f'Precision-Recall Curve (AP = {avg_precision:.2f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.grid(True)
# plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid={
    'n_estimators':[100,200],
    'max_depth':[6,8,10],
    'min_samples_leaf':[1,2]
}

grid_search=GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5, # 5 fold cross-validation
    scoring='accuracy',
    n_jobs=-1, # use all cores
    verbose=1
)

grid_search.fit(x,y)

print('Best Parameters: ', grid_search.best_params_)
print('Best Score: ', grid_search.best_score_)

fianl_model=RandomForestClassifier(max_depth=8, min_samples_leaf=1, n_estimators=200, random_state=42)
fianl_model.fit(x,y) # Train on the full dataset

# import joblib

# joblib.dump(fianl_model,'final_random_forest_model.pkl')

