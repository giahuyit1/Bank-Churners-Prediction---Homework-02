from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

def train_random_forest(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced_subsample",  # tốt hơn cho imbalance
        random_state=42
    )
    model.fit(X, y)
    return model


# ============================================================
# EVALUATE
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n===== RANDOM FOREST PERFORMANCE =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))