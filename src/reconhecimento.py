import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from joblib import load
from src.utils import extrair_vetor, vetor_para_lista

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

MODEL_PATH = "modelos/knn_model.pkl"

def reconhecer(cap):
    if not os.path.exists(MODEL_PATH):
        print("Treine o modelo antes de reconhecer.")
        return

    modelo_salvo = load(MODEL_PATH)
    modelo = modelo_salvo["model"]
    scaler = modelo_salvo["scaler"]

    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            vetor = extrair_vetor(hand)
            vetor_flat = vetor_para_lista(vetor)
            df = pd.DataFrame([vetor_flat], columns=[f"x{i}" for i in range(len(vetor_flat))])
            df_scaled = scaler.transform(df)
            pred = modelo.predict(df_scaled)[0]

            cv2.putText(frame, f"Gesto: {pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Reconhecimento de Gestos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cv2.destroyAllWindows()
