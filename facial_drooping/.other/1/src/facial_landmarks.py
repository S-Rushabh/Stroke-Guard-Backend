import numpy as np
import dlib

def get_landmarks(image, face, predictor):
    landmarks = predictor(image, face)
    coords = np.zeros((68, 2), dtype=int)
    
    for i in range(68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    return coords
