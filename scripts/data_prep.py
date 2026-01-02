import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def main():
    # Load BALANCED dataset
    df = pd.read_csv("data/diabetes_balanced_20k_50_50.csv")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Stratified split (still good practice)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=9)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # Save everything
    joblib.dump(imputer, "models/imputer.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(selector, "models/selector.pkl")
    joblib.dump(X_train, "models/X_train.pkl")
    joblib.dump(X_test, "models/X_test.pkl")
    joblib.dump(y_train, "models/y_train.pkl")
    joblib.dump(y_test, "models/y_test.pkl")

    print("✅ Data preparation completed (50–50 balanced dataset)")

if __name__ == "__main__":
    main()
