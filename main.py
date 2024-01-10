import numpy as np
import cv2

# Paths to the MobileNet SSD model files and minimum confidence threshold
prototext_path = 'models/MobileNetSSD_deploy.prototxt.txt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

# List of classes for the MobileNet SSD model
classes = ["background", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
np.random.seed(543210)
# Generating random colors for each class for visualization
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Reading the MobileNet SSD model using OpenCV
net = cv2.dnn.readNetFromCaffe(prototext_path, model_path)

# Initializing the video capture (using the default camera 0, change it for other cameras)
cap = cv2.VideoCapture(0)

# Continuous loop to grab frames from the camera
while True:
    # Reading a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Getting the frame dimensions
    height, width = frame.shape[:2]

    # Preprocessing the frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()

    # Iterating through detected objects
    for i in range(detected_objects.shape[2]):
        # Checking confidence level for object detection
        confidence = detected_objects[0, 0, i, 2]
        if confidence > min_confidence:
            # Getting class index and bounding box coordinates
            class_index = int(detected_objects[0, 0, i, 1])
            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            # Creating text with the class and confidence
            prediction_text = f"{classes[class_index]}: {confidence:.2f}%"

            # Drawing bounding boxes and labels on the frame
            cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
            cv2.putText(frame, prediction_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

    # Displaying the frame with detected objects
    cv2.imshow("Detected Objects", frame)
    
    # Checking for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the video capture and closing the OpenCV windows
cap.release()
cv2.destroyAllWindows()
