from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def image_ready(name):
    img = load_img(name, target_size=(305, 430))
    array = img_to_array(img)
    array = preprocess_input(array)
    return array
def prediction(name_inp):
    model = load_model("resnet3.h5")
    
    img = image_ready(name_inp)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)
    
    class_labels = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    
    # Get the corresponding class label
    predicted_class_label = class_labels[predicted_class_index]
    
    # Print the predicted class label
    return predicted_class_label