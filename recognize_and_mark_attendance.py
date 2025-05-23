import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

DATASET_PATH = 'dataset'
ATTENDANCE_FILE = 'attendance.csv'

def cargar_faces_referencia():
    base_faces = []
    for persona in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, persona)
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            base_faces.append({'name': persona, 'img_path': img_path})
    return base_faces

# Inicializar cámara y detector
cap = cv2.VideoCapture(0)
referencias = cargar_faces_referencia()
asistentes = set()

# Crear o cargar archivo de asistencia
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Nombre", "Fecha", "Hora"])
    df.to_csv(ATTENDANCE_FILE, index=False)

print("Presiona 'q' para terminar el registro de asistencia.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        for ref in referencias:
            result = DeepFace.verify(frame, ref['img_path'], enforce_detection=False)
            if result['verified'] and ref['name'] not in asistentes:
                now = datetime.now()
                fecha = now.strftime("%Y-%m-%d")
                hora = now.strftime("%H:%M:%S")
                asistentes.add(ref['name'])
                print(f"{ref['name']} registrado a las {hora}")
                df = pd.read_csv(ATTENDANCE_FILE)
                df.loc[len(df.index)] = [ref['name'], fecha, hora]
                df.to_csv(ATTENDANCE_FILE, index=False)
    except Exception as e:
        pass  # puede ocurrir un error si no hay rostros detectados

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
