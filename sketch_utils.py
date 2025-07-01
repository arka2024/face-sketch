import cv2

def convert_to_sketch(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return sketch

def detect_faces(frame):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)