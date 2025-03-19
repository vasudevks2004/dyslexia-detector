from imutils.video import VideoStream
import os
import numpy as np
import imutils
import cv2
import tensorflow as tf

# Possible eye movement labels
list1 = ['looking at center', 'looking at left', 'looking at right', 'looking at up', 'looking at down']

# Load trained CNN model
model_path = 'trained_model_CNN1.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

eye_cnn = tf.keras.models.load_model(model_path)

# Histogram equalization function
def histogram_equalization(img):
    if img is None:
        raise ValueError("Error: Image not loaded properly.")
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    r, g, b = cv2.split(img)
    f_img1 = cv2.equalizeHist(r)
    f_img2 = cv2.equalizeHist(g)
    f_img3 = cv2.equalizeHist(b)
    img = cv2.merge((f_img1, f_img2, f_img3))
    return img

# Function to get index positions
def get_index_positions_2(list_of_elems, element):
    return [i for i, val in enumerate(list_of_elems) if val == element]

# Define model paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir, 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir, 'model_data/weights.caffemodel')

if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
    raise FileNotFoundError("Caffe model files not found! Check 'deploy.prototxt' and 'weights.caffemodel'.")

# Load face detection model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Load eye cascade
eye_cascade_path = 'haar cascade files/haarcascade_eye.xml'
if not os.path.exists(eye_cascade_path):
    raise FileNotFoundError(f"Eye cascade file {eye_cascade_path} not found!")

eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Start video capture
vs = cv2.VideoCapture(0)
eyemovement = []
Dyslexia_result = []
n1, n2 = 0, 10

while True:
    ret, frame = vs.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    frame = imutils.resize(frame, width=750, height=512)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.40:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        f_img = frame[startY:endY, startX:endX]
        if f_img.shape[0] == 0 or f_img.shape[1] == 0:
            continue

        f_img = histogram_equalization(f_img)
        roi_gray = cv2.cvtColor(f_img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(roi_gray)

        pred = None
        for cn, (ex, ey, ew, eh) in enumerate(eyes):
            if cn == 1:
                one_eye = cv2.resize(f_img[ey:ey+eh, ex:ex+ew], (28, 28))  # Resize to (28,28)
                one_eye = cv2.cvtColor(one_eye, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                one_eye = one_eye.astype('float32') / 255.0  # Normalize pixel values
                one_eye = np.expand_dims(one_eye, axis=-1)  # Add channel dimension (28,28,1)
                one_eye = np.expand_dims(one_eye, axis=0)  # Add batch dimension (1,28,28,1)

                # Predict eye movement
                predictions = eye_cnn.predict(one_eye)
                pred = np.argmax(predictions, axis=-1)[0]  # Extract the predicted index

                # Debugging: Print model output
                print(f"Raw predictions: {predictions}")
                print(f"Predicted index: {pred}")

                eyemovement.append(pred)

            cv2.rectangle(f_img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            frame[startY:endY, startX:endX] = f_img
            if cn == 1:
                break

        # Ensure `pred` is within valid range
        if pred is None or pred >= len(list1) or pred < 0:
            print(f"Warning: Invalid prediction index {pred}, defaulting to 'Eyes Not Detected'.")
            text = "Eyes Not Detected"
        else:
            text = list1[pred]

        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 200, 200), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 150), 2)

    cv2.imshow("Frame", frame)

    # Dyslexia detection logic
    if len(eyemovement) >= 10 and len(eyemovement) >= n2:
        eye_array = eyemovement[n1:n2]
        Dyslexia_result.append(1 if len(np.unique(eye_array)) > 2 else 0)
        n1 += 20
        n2 += 20

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()

# Count occurrences of positive and negative results
number_of_positive = get_index_positions_2(Dyslexia_result, 1)
number_of_negative = get_index_positions_2(Dyslexia_result, 0)

if len(number_of_positive) >= 10 or len(number_of_positive) > len(number_of_negative):
    print("Symptoms of Dyslexia detected")
else:
    print("Symptoms of Dyslexia NOT detected")
