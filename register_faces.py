import cv2
import os

# Crear carpeta para almacenar los rostros
DATASET_PATH = 'dataset'
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

nombre = input("Introduce el nombre de la persona: ").strip()
person_path = os.path.join(DATASET_PATH, nombre)

if not os.path.exists(person_path):
    os.makedirs(person_path)

# Captura desde la cámara
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Presiona 'q' para salir una vez se hayan capturado suficientes imágenes.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = frame[y:y+h, x:x+w]
        cv2.imwrite(f"{person_path}/{count}.jpg", rostro)
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Capturando Rostros', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Se han guardado {count} imágenes en {person_path}")
