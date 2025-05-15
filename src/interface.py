import tkinter as tk
from tkinter import ttk, simpledialog
from src.coleta_dados import coletar_dados
from src.treino import treinar
from src.reconhecimento import reconhecer

class App(tk.Tk):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.title("Reconhecimento de Gestos")
        self.geometry("380x310")
        self.configure(bg="#2C2F33")

        style = ttk.Style(self)
        style.theme_use('clam')

        style.configure('TButton',
                        background='#5865F2',
                        foreground='white',
                        font=('Segoe UI', 14, 'bold'),
                        padding=10,
                        borderwidth=0)
        style.map('TButton',
                  background=[('active', '#4752C4')],
                  relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        style.configure('Exit.TButton',
                        background='#23272A',
                        foreground='white',
                        font=('Segoe UI', 12),
                        padding=10,
                        borderwidth=0)
        style.map('Exit.TButton',
                  background=[('active', '#1A1D21')],
                  relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        frame = ttk.Frame(self, padding=20)
        frame.pack(expand=True, fill='both')
        frame.configure(style="Dark.TFrame")

        style.configure("Dark.TFrame", background="#2C2F33")

        # Botões
        self.btn_coletar = ttk.Button(frame, text="Coletar Dados", command=self.coletar)
        self.btn_coletar.pack(fill='x', pady=10)

        self.btn_treinar = ttk.Button(frame, text="Treinar Modelo", command=self.treinar)
        self.btn_treinar.pack(fill='x', pady=10)

        self.btn_reconhecer = ttk.Button(frame, text="Reconhecer Gestos", command=self.reconhecer)
        self.btn_reconhecer.pack(fill='x', pady=10)

        self.btn_sair = ttk.Button(frame, text="Sair", command=self.sair, style='Exit.TButton')
        self.btn_sair.pack(fill='x', pady=15)

    def coletar(self):
        label = simpledialog.askstring("Coletar Dados", "Nome do gesto:", parent=self)
        if label:
            coletar_dados(label.lower().strip(), self.cap)

    def treinar(self):
        treinar()

    def reconhecer(self):
        reconhecer(self.cap)

    def sair(self):
        self.cap.release()
        self.destroy()
