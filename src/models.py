"""
PathSnap Model Experimentation & Evaluation
Module for Member 2: Training, cross-validation, and comparing 5 model families.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def get_model_registry() -> Dict[str, Any]:
    """
    Returns baseline models covering the required algorithm families:
    1. Linear/Probabilistic
    2. Instance-based
    3. Tree/Ensemble
    4. Gradient Boosting
    5. Neural Network (MLP)
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }
