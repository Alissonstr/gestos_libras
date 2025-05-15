import os
import shutil
from datetime import datetime
import pandas as pd
import numpy as np

DATA_FILE = "data/gestos.csv"
BACKUP_FOLDER = "data/backups"

def extrair_vetor(landmarks):
    return [(p.x, p.y, p.z) for p in landmarks.landmark]

def vetor_para_lista(vetor):
    return np.array(vetor).flatten()

def salvar_dados(vetor, label):
    vetor_flat = vetor_para_lista(vetor)
    dados = np.append(vetor_flat, label)
    df = pd.DataFrame([dados])
    colunas = [f"x{i}" for i in range(len(vetor_flat))] + ["label"]

    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, mode="w", header=colunas, index=False)
    else:
        df.to_csv(DATA_FILE, mode="a", header=False, index=False)

def criar_backup():
    if not os.path.exists(DATA_FILE):
        print("Sem arquivo para backup.")
        return

    os.makedirs(BACKUP_FOLDER, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_FOLDER, f"gestos_backup_{ts}.csv")
    shutil.copy(DATA_FILE, backup_path)
    print(f"Backup criado: {backup_path}")
