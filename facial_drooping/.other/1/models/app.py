import tensorflow as tf

# Load the .dat file (assuming it's a serialized model)
model = tf.keras.models.load_model("shape_predictor_68_face_landmarks.dat")

# Convert to .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
