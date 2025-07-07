import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(path="C:/Users/LENOVO/OneDrive/Documents/shubham-personal/OneDrive/Desktop/git_projects/day2_titanic_analysis/train.csv"):
    df = pd.read_csv(path)

    # Drop irrelevant columns
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

    # Drop rows with missing Age or Embarked
    df.dropna(subset=["Age", "Embarked"], inplace=True)

    # Encode categorical features
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    X = df.drop(columns=["Survived", "PassengerId"])
    y = df["Survived"]

    return X, y

def select_k_best_features(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()]
    scores = selector.scores_

    print("\n🎯 Top features by SelectKBest:")
    for col, score in zip(X.columns, scores):
        print(f"{col}: {score:.2f}")

    # Plotting scores
    plt.figure(figsize=(10, 5))
    sns.barplot(x=X.columns, y=scores)
    plt.xticks(rotation=0)
    plt.title("SelectKBest F-scores")
    plt.tight_layout()
    plt.savefig("selectkbest_scores.png")
    plt.show()

    return selected

def select_rfe_features(X, y, n_features=5):
    model = RandomForestClassifier(random_state=42)
    selector = RFE(estimator=model, n_features_to_select=n_features)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()]

    print("\n🎯 Top features by RFE:")
    print(selected.tolist())

    return selected

if __name__ == "__main__":
    print("📦 Loading and cleaning data...")
    X, y = load_and_prepare_data()

    print("\n🔍 Running SelectKBest...")
    kbest_selected = select_k_best_features(X, y)

    print("\n🧠 Running RFE...")
    rfe_selected = select_rfe_features(X, y)

    print("\n✅ Done. Plot saved as 'selectkbest_scores.png'")
