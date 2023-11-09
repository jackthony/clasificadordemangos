# clasificadormangos.py

from keras.models import load_model
import cv2
import numpy as np

cv2.startWindowThread()
np.set_printoptions(suppress=True)

def run_classification():
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, image = camera.read()

        if not ret:
            print("Error al capturar la imagen desde la cámara.")
            break

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imshow("Webcam Image", image)

        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        if confidence_score > 0.90:
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        else:
            print("Predicción no válida.")

        keyboard_input = cv2.waitKey(1)

        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_classification()
