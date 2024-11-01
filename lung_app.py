import streamlit as st
import cv2
import numpy as np
import pickle


# defining a function to Load the Model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image):
    # Reading the image using OpenCV
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resizing the image to match the size used during training
    image_resized = cv2.resize(image_gray, (128, 128))
    
    # Flatten the resized image into a 1D array
    image_flat = image_resized.reshape(1, -1)
    
    return image_flat

def main():
    st.title("Lung Disease Classifier")
    
    # Load the trained model
    model_path = 'Random_Forest_Classifier.pkl'
    model = load_model(model_path)
    
    # Prompting the user to upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        image_flat = preprocess_image(image)
        
        # Make predictions
        prediction = model.predict(image_flat)
        
        # Print the predicted label
        st.write(f'Predicted label: {prediction[0]}')

if __name__ == "__main__":
    main()
