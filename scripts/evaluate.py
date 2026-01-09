import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

def main():
    # Load model and test data
    model = joblib.load("models/best_model.pkl")
    X_test = joblib.load("models/X_test.pkl")
    y_test = joblib.load("models/y_test.pkl")

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Threshold
    THRESHOLD = 0.5
    y_pred = (probs >= THRESHOLD).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)

    print("📈 Final Evaluation (Balanced Dataset)")
    print("Threshold :", THRESHOLD)
    print("Accuracy  :", acc)
    print("Precision :", prec)
    print("Recall    :", rec)
    print("F1 Score  :", f1)
    print("ROC-AUC   :", auc)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -------- ROC CURVE --------
    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Diabetes Prediction Model")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
