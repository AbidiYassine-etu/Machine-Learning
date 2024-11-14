import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek  # Correction ici
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Charger les données
df = pd.read_csv('gym_members_exercise_tracking.csv')

# Nettoyage des colonnes
df.columns = df.columns.str.strip()

# Définir les caractéristiques et la cible
target_column = 'Workout_Type'
X = df.drop(columns=[target_column])
y = df[target_column]

# Convertir les étiquettes cibles en valeurs numériques
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Définir les colonnes catégorielles et numériques
categorical_columns = ['Gender', 'Experience_Level']
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

# Prétraitement des données
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_columns),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_columns)
    ])

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Appliquer le prétraitement sur les données d'entraînement et de test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Appliquer SMOTETomek pour gérer les classes déséquilibrées
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train_processed, y_train)

# Paramètres de recherche pour Random Forest
param_grid_rf = {
    'n_estimators': [300],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Initialiser GridSearchCV pour Random Forest
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_resampled, y_resampled)

# Paramètres de recherche pour XGBoost
param_grid_xgb = {
    'n_estimators': [200],
    'max_depth': [6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialiser GridSearchCV pour XGBoost
grid_search_xgb = GridSearchCV(estimator=xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
                               param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_resampled, y_resampled)

# Initialiser le modèle GradientBoostingClassifier
param_grid_gb = {
    'n_estimators': [200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}

grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                              param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_resampled, y_resampled)

# Créer le VotingClassifier pour combiner les modèles
voting_clf = VotingClassifier(estimators=[
    ('rf', grid_search_rf.best_estimator_),
    ('xgb', grid_search_xgb.best_estimator_),
    ('gb', grid_search_gb.best_estimator_)
], voting='soft')

# Entraîner le VotingClassifier sur les données rééchantillonnées
voting_clf.fit(X_resampled, y_resampled)

# Prédictions sur l'ensemble de test
y_pred_voting = voting_clf.predict(X_test_processed)

# Calculer l'accuracy pour le VotingClassifier
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f'Voting Classifier Accuracy: {accuracy_voting * 100:.2f}%')

# Validation croisée pour évaluer le VotingClassifier
cv_scores_voting = cross_val_score(voting_clf, X_resampled, y_resampled, cv=5)
print(f'Voting Classifier Cross-Validation Accuracy: {cv_scores_voting.mean() * 100:.2f}%')
