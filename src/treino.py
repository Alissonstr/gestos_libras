import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from src.utils import criar_backup, DATA_FILE

MODEL_PATH = "modelos/knn_model.pkl"

def treinar():
    if not os.path.exists(DATA_FILE):
        print("Não há dados para treinar.")
        return

    criar_backup()

    df = pd.read_csv(DATA_FILE)
    X = df.drop("label", axis=1)
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=9, weights="distance")
    knn.fit(X_train, y_train)

    acc = knn.score(X_test, y_test)
    print(f"Acurácia: {acc*100:.2f}%")

    os.makedirs("modelos", exist_ok=True)
    dump({"model": knn, "scaler": scaler}, MODEL_PATH)
    print("Modelo salvo.")
