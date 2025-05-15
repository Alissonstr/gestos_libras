import cv2
from src.interface import App

def main():
    cap = cv2.VideoCapture("http://192.168.1.4:8080/video")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)

    app = App(cap)
    app.mainloop()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
