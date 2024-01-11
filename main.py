import numpy as np
import cv2

# Mes Paths et le min_conf
prototext_path = 'models/MobileNetSSD_deploy.prototxt.txt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.5

# lites des classes
classes = ["background", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
np.random.seed(543210)
# generer les couleurs
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# lire le MobilNet with using open cv
net = cv2.dnn.readNetFromCaffe(prototext_path, model_path)

# initialiser cam 
cap = cv2.VideoCapture(0)

# La boucle pour detecter
while True:
    # detecte frame from the cam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # prendre les mesures d'un frame
    height, width = frame.shape[:2]

    # rendre le frame mesurable pour detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()

    # itteration. pour la detection
    for i in range(detected_objects.shape[2]):
        # chekch the min confi
        confidence = detected_objects[0, 0, i, 2]
        if confidence > min_confidence:
            # comparer avec les classes compatible
            class_index = int(detected_objects[0, 0, i, 1])
            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            # creation du texte
            prediction_text = f"{classes[class_index]}: {confidence:.2f}%"

            # dessiner le rect et le texte
            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
            cv2.putText(frame, prediction_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

    # afficher les ob
    cv2.imshow("Detected Objects", frame)
    
  
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# closing open cv 
cap.release()
cv2.destroyAllWindows()