
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('gym_members_exercise_tracking.csv')
data_encoded = pd.get_dummies(data.drop('Workout_Type', axis=1), drop_first=True)

X = data_encoded
y = data['Workout_Type']  


from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model1 = RandomForestClassifier(random_state=42)
model2 = SVC(kernel='linear', random_state=42)
model3 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

ensemble_model = VotingClassifier(estimators=[('rf', model1), ('svc', model2), ('mlp', model3)], voting='hard')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=model1, param_grid=param_grid, cv=5, verbose=3)
grid_search.fit(X_train_scaled, y_train_resampled)

best_rf_model = grid_search.best_estimator_

y_pred = best_rf_model.predict(X_test_scaled)

print("Best Model from GridSearch:", best_rf_model)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

ensemble_model.fit(X_train_scaled, y_train_resampled)
y_pred_ensemble = ensemble_model.predict(X_test_scaled)
print("Ensemble Model Accuracy on Test Set:", accuracy_score(y_test, y_pred_ensemble))
print("Ensemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train_resampled, cv=5, scoring='accuracy')
print("Cross-validation Scores:", cv_scores)
print("Mean Cross-validation Score:", np.mean(cv_scores))
accuracy_rf = accuracy_score(y_test, y_pred_ensemble)


print(f'Accuracy: {accuracy_rf * 100:.2f}%')


# L'algorithme "Ensemble Learning avec Grid Search et SMOTE". Voici une décomposition plus précise des algorithmes utilisés :

# Random Forest Classifier (for Classification) : Un algorithme d'ensemble basé sur des arbres de décision.
# Support Vector Classifier (SVC) : Un classificateur basé sur des machines à vecteurs de support, qui cherche à séparer les classes avec l'hyperplan optimal.
# Multi-layer Perceptron (MLPClassifier) : Un réseau de neurones à plusieurs couches.
# Voting Classifier : Un classificateur d'ensemble qui combine les prédictions de plusieurs modèles en utilisant un vote majoritaire.
# SMOTE (Synthetic Minority Over-sampling Technique) : Une technique de suréchantillonnage pour équilibrer les classes déséquilibrées.
# Grid Search with Cross Validation : Une méthode d'optimisation des hyperparamètres qui utilise la recherche exhaustive et la validation croisée pour trouver les meilleurs paramètres pour un modèle.