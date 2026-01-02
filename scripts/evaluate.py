import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    model = joblib.load("models/best_model.pkl")
    X_test = joblib.load("models/X_test.pkl")
    y_test = joblib.load("models/y_test.pkl")

    probs = model.predict_proba(X_test)[:, 1]

    THRESHOLD = 0.5
    y_pred = (probs >= THRESHOLD).astype(int)

    print("📈 Final Evaluation (Balanced Dataset)")
    print("Threshold:", THRESHOLD)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
