from gtts import gTTS
import pygame
import time
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#for test time augmentation
test_datagen = ImageDataGenerator(  rescale = 1./255,
                                    horizontal_flip=True,
                                    rotation_range = 20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   vertical_flip=True,                                   
                                   zoom_range = 0.2,
                                   )


def main():
    st.title("Nepali currency Classifier web app")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for the deep learning model
        input_array = preprocess_frame(image)
        
        # Load the model
        model = load_model('keras_models/new_final_2.h5')  # Replace with your model path

        # Make predictions
        predictions = model.predict_generator(input_array)
        summed = np.sum(predictions,axis =0)
        predicted_class = np.argmax(summed)
        label = get_label(predicted_class)
        text_to_speech(label)
        prediction_label = f'Prediction: {label}'
        play_audio()
        st.write(prediction_label)

def preprocess_frame(frame):
    
    target_size = (512,512)  
    
    frame = cv2.resize(frame, target_size)
  
    frame = np.array(frame)
    
    frames = np.expand_dims(frame,0)
    

    return test_datagen.flow(frames,batch_size = 2)

def get_label(class_index):
    labels = ['fifty','five','five hundred','hundred','ten','thousand','twenty']  # Replace with your class labels
    return labels[class_index]

def text_to_speech(text, language='en', filename='output.mp3'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(filename)

def play_audio(filename='output.mp3'):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    time.sleep(5)

if __name__ == "__main__":
    main()
