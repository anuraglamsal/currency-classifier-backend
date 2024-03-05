import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#for test time augmentation
test_datagen = ImageDataGenerator(rescale = 1./255,
                                  horizontal_flip=True,
                                  rotation_range = 20,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  featurewise_center=True,
                                  featurewise_std_normalization=True,
                                  vertical_flip=True,                                   
                                  zoom_range = 0.2,
                                  )

model = load_model('keras_models/new_final.h5')  # Replace with your model path

def predict(fileName):

    # Read the image
    image = cv2.imread(f'uploads/{fileName}');
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess the image for the deep learning model
    input_array = preprocess_frame(image)

    # Make predictions
    predictions = model.predict_generator(input_array)
    summed = np.sum(predictions, axis=0)
    predicted_class = np.argmax(summed)
    label = get_label(predicted_class)

    return label 

def preprocess_frame(frame):
    
    target_size = (512, 512)  

    frame = cv2.resize(frame, target_size)
    frame = np.array(frame)
    frames = np.expand_dims(frame,0)

    return test_datagen.flow(frames, batch_size=2)

def get_label(class_index):

    labels = ['fifty','five','five hundred','hundred','ten','thousand','twenty']  # Replace with your class labels

    return labels[class_index]
