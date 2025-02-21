import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Load TFLite model
tflite_model_path = 'facial_paralysis_detector.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess image
def preprocess_image(image, target_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# Function to detect facial drooping
def detect_drooping(image):
    input_data = preprocess_image(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    
    # Assuming output > 0.5 means drooping detected
    if result[0][0] > 0.5:
        return "Facial Drooping Detected"
    else:
        return "No Facial Drooping"

# Capture and analyze image
def capture_image():
    ret, frame = cap.read()
    if ret:
        result = detect_drooping(frame)
        messagebox.showinfo("Detection Result", result)
        
        # Show captured image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        captured_label.configure(image=img_tk)
        captured_label.image = img_tk

# Initialize main window
root = tk.Tk()
root.title("Facial Drooping Detection")
root.geometry("800x600")

# Create capture button
capture_button = tk.Button(root, text="Capture", command=capture_image, font=("Arial", 20), bg="blue", fg="white")
capture_button.pack(pady=20)

# Label to display captured image
captured_label = tk.Label(root)
captured_label.pack()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to continuously display the webcam feed
def show_webcam():
    ret, frame = cap.read()
    if ret:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        captured_label.configure(image=img_tk)
        captured_label.image = img_tk
    root.after(10, show_webcam)

show_webcam()

# Run the app
root.mainloop()

# Release the webcam
cap.release()
cv2.destroyAllWindows()
