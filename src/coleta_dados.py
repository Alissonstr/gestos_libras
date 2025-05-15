import cv2
import mediapipe as mp
from src.utils import salvar_dados, extrair_vetor

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def coletar_dados(label, cap):
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("Pressione 'c' para capturar gesto, 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler câmera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]  # só a primeira mão
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Coletando: {label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Coleta de Dados", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break

        if k == ord('c') and res.multi_hand_landmarks:
            vetor = extrair_vetor(hand)
            salvar_dados(vetor, label)

    hands.close()
    cv2.destroyAllWindows()
