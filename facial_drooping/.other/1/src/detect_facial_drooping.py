import cv2
import dlib
import numpy as np
from facial_landmarks import get_landmarks
from asymmetry_analysis import analyze_asymmetry

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/Project/Stroke-Guard/backend/facial_drooping/models/shape_predictor_68_face_landmarks.dat")

def detect_facial_drooping(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No face detected.")
        return

    for face in faces:
        landmarks = get_landmarks(gray, face, predictor)

        # Draw landmarks on image
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Analyze facial drooping
        asymmetry_score, is_drooping = analyze_asymmetry(landmarks)

        # Display results dynamically based on real asymmetry score
        result_text = "Facial Drooping Detected" if is_drooping else "No Drooping"
        color = (0, 0, 255) if is_drooping else (0, 255, 0)

        cv2.putText(image, f"Asymmetry Score: {asymmetry_score:.4f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, result_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Save and display result
    output_path = "../results/detected_" + image_path.split("/")[-1]
    cv2.imwrite(output_path, image)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python detect_facial_drooping.py --image <image_path>")
    else:
        detect_facial_drooping(sys.argv[2])
