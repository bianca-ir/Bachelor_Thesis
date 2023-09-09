
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
import numpy as np
import preprocessing.gaussian_blur
import preprocessing.clahe 
import cv2
import numpy as np
import os
import feature_extraction.get_intensity
import feature_extraction.get_GLCM
import pickle
from tensorflow.keras.models import model_from_json



app = Flask(__name__)
app.config['LOGGER_HANDLER_POLICY'] = 'always'

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload(): 
    if 'file' not in request.files:
        return 'No file uploaded', 400


    file = request.files['file']
    saved_filename = file.filename
    
    save_directory = 'app/uploads'

    
    file_path = os.path.join(save_directory, saved_filename)
    file.save(file_path)

    return jsonify({'message': 'File uploaded successfully', 'filename': saved_filename})


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    if 'filename' not in data:
        print('No filename provided')
    parameters = data['parameters']
    filename = data['filename']

    selected_model = parameters['algorithm']


    if selected_model == 'rf':
        
        selected_configuration = parameters['configuration']

        if selected_configuration == 'best':
            with open('app/saved_models/rf_model_first.pkl', 'rb') as file:
                rf_model = pickle.load(file)
        else:
            with open('app/saved_models/rf_model_second.pkl', 'rb') as file:
                rf_model = pickle.load(file)


        image = cv2.imread('app/uploads/' + filename, cv2.IMREAD_GRAYSCALE)
        

        
        processed_image = preprocessing.gaussian_blur.get_Gaussian_blur(image)
        features = feature_extraction.get_intensity.extract_intensity_single(processed_image)

        
       
        reshaped_features = features.reshape(1, -1)


         
        class_probabilities = rf_model.predict_proba(reshaped_features)
        predicted_class = rf_model.classes_[np.argmax(class_probabilities)]
        benign_prob = class_probabilities[0, 0]
        malignant_prob = class_probabilities[0, 1]


       
        response = {
            'predicted_class': predicted_class,
            'benign_probability': benign_prob,
            'malignant_probability': malignant_prob
        }

          

    elif selected_model == 'svm': 
           
        selected_configuration = parameters['configuration']

        if selected_configuration == 'best':
            with open('app/saved_models/svm_model_first.pkl', 'rb') as file:
                svm_model = pickle.load(file)
        else:
            with open('app/saved_models/svm_model_second.pkl', 'rb') as file:
                svm_model = pickle.load(file)


        image = cv2.imread('app/uploads/' + filename, cv2.IMREAD_GRAYSCALE)
      
       
        processed_image = preprocessing.clahe.get_CLAHE(image)
        
        features = feature_extraction.get_GLCM.extract_GLCM_single(image)

       
        reshaped_features = features.reshape(1, -1)

    
        
        class_probabilities = svm_model.predict_proba(reshaped_features)
        predicted_class = svm_model.classes_[np.argmax(class_probabilities)]
        benign_prob = class_probabilities[0, 0]
        malignant_prob = class_probabilities[0, 1]


       
        response = {
            'predicted_class': predicted_class,
            'benign_probability': benign_prob,
            'malignant_probability': malignant_prob
        }
    
    
    else: 
        img = cv2.imread('app/uploads/' + filename, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(img, (400, 256))
        normalized_image = resized_image / 255.0
        input_image = np.reshape(normalized_image, (1, 256, 400, 1))

     
        selected_configuration = parameters['configuration']

        if selected_configuration == 'best':
            with open('app/saved_models/cnn_architecture.pkl', 'rb') as file:
                model_architecture = pickle.load(file)
            with open('app/saved_models/cnn_weights.pkl', 'rb') as file:
                model_weights = pickle.load(file)
        else:
            with open('app/saved_models/cnn_architecture_second.pkl', 'rb') as file:
                model_architecture = pickle.load(file)
            with open('app/saved_models/cnn_weights_second.pkl', 'rb') as file:
                model_weights = pickle.load(file)


       
        cnn_model = model_from_json(model_architecture)
        cnn_model.set_weights(model_weights)

        
        class_probabilities = cnn_model.predict(input_image)

     
        predicted_class = np.argmax(class_probabilities)
        benign_prob = float(class_probabilities[0, 0])
        malignant_prob = float(class_probabilities[0, 1])

        
       
        if predicted_class == 0:
            predicted_class_label = "Benign"
        else:
            predicted_class_label = "Malignant"

  

        response = {
            'predicted_class': predicted_class_label,
            'benign_probability': benign_prob,
            'malignant_probability': malignant_prob
        }

    
    return jsonify(response)

 

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=0)

    
