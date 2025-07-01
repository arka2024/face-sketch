import cv2
import time
import os
from sketch_utils import convert_to_sketch, detect_faces  # Fixed import path

# Initialize directories
if not os.path.exists("outputs"):  
    os.makedirs("outputs")  

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Auto-Sketching Machine Active! Press:")
    print("- 's' to save current sketch")
    print("- 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture frame")
            break
        
        faces = detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            sketch_face = convert_to_sketch(face_roi)
            frame[y:y+h, x:x+w] = cv2.cvtColor(sketch_face, cv2.COLOR_GRAY2BGR)
        
        cv2.imshow('Auto-Sketch Machine', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = int(time.time())
            cv2.imwrite(f"outputs/sketch_{timestamp}.jpg", sketch_face)
            print(f"Sketch saved as outputs/sketch_{timestamp}.jpg")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()