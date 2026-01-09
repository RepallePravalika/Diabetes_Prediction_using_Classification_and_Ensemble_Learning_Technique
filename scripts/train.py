import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

def main():
    print("🚀 Training started...")

    X_train = joblib.load("models/X_train.pkl")
    X_test = joblib.load("models/X_test.pkl")
    y_train = joblib.load("models/y_train.pkl")
    y_test = joblib.load("models/y_test.pkl")

    base_models = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("svm", SVC(probability=True)),
        ("rf", RandomForestClassifier(n_estimators=150, random_state=42)),
        ("gb", GradientBoostingClassifier())
    ]

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X_train, y_train, cv=cv)

    print(f"📊 Cross-Validation Accuracy: {scores.mean():.4f}")

    stack.fit(X_train, y_train)

    preds = stack.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Test Accuracy (threshold=0.5): {acc:.4f}")

    joblib.dump(stack, "models/best_model.pkl")
    print("💾 Model saved")

if __name__ == "__main__":
    main()
