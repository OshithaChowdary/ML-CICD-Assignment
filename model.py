"""Train and evaluate a simple Iris classification model."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
    """Train the model and return (model, accuracy)."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


def main():
    iris = load_iris()
    model, accuracy = train_model()
    sample_prediction = model.predict([iris.data[0]])[0]
    print("ML Model: Random Forest Classifier")
    print("Dataset: Iris")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Sample prediction: {iris.target_names[sample_prediction]}")


if __name__ == "__main__":
    main()
