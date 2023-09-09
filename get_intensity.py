
import numpy as np


from skimage.feature import graycomatrix, graycoprops


def extract_intensity_single(img):          
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    image_features = []
    features = []

    for distance in distances:
        for angle in angles:
            std = np.std(img)
            mean = np.mean(img)
            kurtosis = np.mean(np.power((img - mean) / np.std(img), 4))
            skewness = np.mean(np.power((img - mean) / np.std(img), 3))
            glcm = graycomatrix(img, distances=[distance], angles=[angle], symmetric=True, normed=True)
 
            contrast = graycoprops(glcm, prop='contrast')[0, 0]
            correlation = graycoprops(glcm, prop='correlation')[0, 0]
            energy = graycoprops(glcm, prop='energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
            image_features.extend([std, mean, kurtosis, skewness, contrast, correlation, energy, homogeneity])


    features.append(image_features)
         

    X = np.array(features)
    return X