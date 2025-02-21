import numpy as np

def analyze_asymmetry(landmarks):
    """
    Computes facial asymmetry by comparing key features on the left and right sides.
    Returns an asymmetry score (higher = more asymmetry).
    """

    # Extract key facial landmarks
    left_eye = landmarks[36:42]   # Left eye
    right_eye = landmarks[42:48]  # Right eye
    mouth = landmarks[48:68]      # Mouth
    left_eyebrow = landmarks[17:22]  # Left eyebrow
    right_eyebrow = landmarks[22:27] # Right eyebrow
    jaw = landmarks[0:17]  # Jawline

    # Compute vertical distances
    left_eye_height = np.mean(left_eye[:, 1])
    right_eye_height = np.mean(right_eye[:, 1])
    left_eyebrow_height = np.mean(left_eyebrow[:, 1])
    right_eyebrow_height = np.mean(right_eyebrow[:, 1])
    left_mouth_corner = mouth[0]
    right_mouth_corner = mouth[6]

    # Calculate differences in vertical positions
    eye_asymmetry = abs(left_eye_height - right_eye_height)
    eyebrow_asymmetry = abs(left_eyebrow_height - right_eyebrow_height)
    mouth_asymmetry = abs(left_mouth_corner[1] - right_mouth_corner[1])

    # Normalize asymmetry by dividing by face width (for scale invariance)
    face_width = jaw[-1][0] - jaw[0][0]  # Distance between far-left and far-right jaw points
    total_asymmetry = (eye_asymmetry + eyebrow_asymmetry + mouth_asymmetry) / face_width

    # Define a dynamic threshold for facial drooping detection
    DROOPING_THRESHOLD = 0.05  # Adjust this based on dataset evaluation

    return total_asymmetry, total_asymmetry > DROOPING_THRESHOLD
