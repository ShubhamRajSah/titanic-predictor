# 🛳 Titanic: Survival Prediction – Machine Learning Case Study

A complete end-to-end data science project exploring the Titanic dataset using Python and scikit-learn.  
This project walks through how to *clean, visualize, engineer features, train models*, and evaluate them to predict passenger survival.

Instead of chasing accuracy alone, this pipeline focuses on building *interpretable, reusable components* — from ml_starter_script.py to the final predictions and .pkl model export. It's built to scale and adapt across datasets.

---
## 🧹 1. Data Preprocessing & Cleaning

Before modeling, several essential transformations were made to address missing values and standardize features:

- *Missing Values Handled:*
  - Age: Filled using the overall median.
  - Embarked: Imputed with the mode 'S'.
  - Fare: Filled in test set (if applicable) using median.
  - Cabin: Used the first character (deck), and missing values were assigned 'M' (for missing).

- *Derived Columns (Before Feature Engineering):*
  - Deck: Extracted from the first letter of Cabin.
  - TicketPrefix: Extracted from ticket using regex, filled with 'X' for missing.

- *Feature Engineering:*
  - FamilySize: Sum of SibSp + Parch + 1
  - IsAlone: Binary flag if passenger is alone (FamilySize == 1)
  - Title: Extracted from Name and grouped rare titles into 'Rare'

- *Encoding:*
  - Applied LabelEncoder to: Sex, Embarked, Title, TicketPrefix, and Deck

- *Standardization:*
  - Used StandardScaler on Age and Fare to normalize distributions

All of this logic was structured via two modular functions: engineer_features(df) and preprocess(df), to keep the pipeline clean and reusable.


---  


## 📊 2. Feature Engineering & Exploratory Data Analysis (EDA)

### 🧠 Feature Engineering Highlights

To enhance model performance and uncover hidden signals, several engineered features were added:

- **FamilySize**: Combines SibSp and Parch → FamilySize = SibSp + Parch + 1  
  > Small to mid-sized families showed better survival odds than solo travelers or large families.

- **IsAlone**: Boolean flag (1 if FamilySize == 1, else 0)  
  > Solo travelers had noticeably lower survival rates in 3rd class.

- **Title**: Extracted from Name using regex (e.g., Mr, Miss, Master)  
  > Rare titles (like “Lady”, “Don”, “Dr”) were grouped into a single "Rare" category.

- **Deck**: Extracted from Cabin's first character — missing entries filled as 'M'  
  > Helped capture the correlation between deck location, fare, and class.

- **TicketPrefix**: Derived from Ticket using regex  
  > Encoded grouping or port-related patterns embedded in ticket codes.

### 🔍 Exploratory Data Analysis (EDA)

Visual exploration using Seaborn and Matplotlib revealed strong trends:

- *Bar Plots*:  
  - Showed significant survival gaps based on Sex, Pclass, and Embarked
  - Women in 1st and 2nd class had the highest survival rates

- *Box Plots*:  
  - Revealed that survivors generally paid higher fares and fell between the ages of 20–40
  - Outliers handled using scaling or binning strategies

- *Correlation Heatmaps*:  
  - Highlighted strong positive survival signals from Fare, Title, and Sex

- *Feature Importance (from Random Forest)*:  
  - Top predictive features: Sex, Title, Fare, and FamilySize

> 📌 Optional: Add .png charts to a local /images folder and embed them with:  
> ![Alt text](images/survival_by_pclass.png)

---

#### 🤖 3. Modeling & Evaluation

#### 🔍 Model Selection

To assess different algorithms, the project tested:

- *Logistic Regression*  
  - A simple, interpretable model used as a performance baseline.

- *Decision Tree*  
  - Handled non-linearity but slightly prone to overfitting.

- *Random Forest*  
  - Best-performing model; reduced variance and improved generalization.
  - Offered built-in feature importance insights.

#### 🧪 Evaluation Strategy

The dataset was split using train_test_split (80% train, 20% test). Key metrics used:

- *Accuracy*: Overall correctness
- *Precision, Recall, F1-score*: Balanced evaluation for both classes
- *Confusion Matrix*: Detailed breakdown of predicted vs. actual labels
- *Precision-Recall Curve*: Visualized trade-offs for imbalanced class handling
 
```python
RandomForestClassifier(n_estimators=100, random_state=42)

---
🧠 4. Best Model Insights & Hyperparameter Tuning

Used GridSearchCV for hyperparameter tuning:

`python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 8, 10],
    'minsamplesleaf': [1, 2]
}
`

`python
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    paramgrid=paramgrid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)
`

✅ Best Results

- Best Parameters: {'maxdepth': 8, 'minsamplesleaf': 1, 'nestimators': 200}
- Best Accuracy (CV): ~82%

Final model trained on full dataset:

`python
final_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    minsamplesleaf=1,
    random_state=42
)

final_model.fit(X, y)
`

💾 Model Export

Saved with:

`python
import joblib
joblib.dump(finalmodel, 'finalrandomforestmodel.pkl')
`

---

🧾 5. Learnings & Reflections

- Gained end-to-end experience in structuring a modular ML pipeline
- Practiced imputation, encoding, and feature extraction under real-world constraints
- Strengthened skills in comparative modeling and evaluation beyond accuracy
- Explored the importance of social/demographic features in survival prediction

---

🚀 6. Future Improvements

- 🌐 Deploy the model via Streamlit to visualize predictions
- 📊 Add real-time visual dashboards (e.g., SHAP explanations)
- 🔁 Add stratified cross-validation throughout
- 📦 Package mlstarterscript.py as a CLI or notebook template

---

📂 Repository Structure

`bash
├── mlstarterscript.py         # Full ML pipeline
├── train.csv                    # Input data
├── finalrandomforest_model.pkl # Saved final model
├── README.md                    # Project documentation (this file)
`

---

💬 Author

Made with focus and flair by Shubham Raj Sah  
Let’s connect on GitHub, collaborate on data projects, or share notes on machine learning!

---
`

