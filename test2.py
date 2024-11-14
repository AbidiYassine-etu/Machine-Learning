import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from imblearn.combine import SMOTETomek

# Load and clean data
df = pd.read_csv('gym_members_exercise_tracking.csv')
df.columns = df.columns.str.strip()
target_column = 'Workout_Type'
X = df.drop(columns=[target_column])
y = df[target_column]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define categorical and numerical columns
categorical_columns = ['Gender', 'Experience_Level']
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numeric_columns),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_columns)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Handle class imbalance with SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train_processed, y_train)

# Model parameters
param_grid_rf = {
    'n_estimators': [300],
    'max_depth': [10, 15, None]
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10],
    'learning_rate': [0.05, 0.1]
}

param_grid_gb = {
    'n_estimators': [200],
    'max_depth': [5],
    'learning_rate': [0.1]
}

param_grid_lgb = {
    'n_estimators': [200],
    'max_depth': [7, 9],
    'learning_rate': [0.05, 0.1]
}

# Initialize models with best parameters
rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1, verbose=1)
xgb_model = GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), param_grid_xgb, cv=3, n_jobs=-1, verbose=1)
gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3, n_jobs=-1, verbose=1)
lgb_model = GridSearchCV(lgb.LGBMClassifier(random_state=42), param_grid_lgb, cv=3, n_jobs=-1, verbose=1)

# Train models
rf.fit(X_resampled, y_resampled)
xgb_model.fit(X_resampled, y_resampled)
gb.fit(X_resampled, y_resampled)
lgb_model.fit(X_resampled, y_resampled)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', rf.best_estimator_),
        ('xgb', xgb_model.best_estimator_),
        ('gb', gb.best_estimator_),
        ('lgb', lgb_model.best_estimator_)
    ],
    final_estimator=LogisticRegression()
)

# Train stacking classifier
stacking_clf.fit(X_resampled, y_resampled)

# Evaluate model performance
y_pred = stacking_clf.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f'Stacking Classifier Accuracy: {accuracy * 100:.2f}%')

# Cross-validation score
cv_scores = cross_val_score(stacking_clf, X_resampled, y_resampled, cv=5)
print(f'Stacking Classifier Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%')
