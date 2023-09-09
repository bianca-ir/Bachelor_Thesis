
from skimage.feature import graycomatrix, graycoprops
import numpy as np 


def extract_GLCM_single(img): 
            distances = [1, 3, 5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            image_features = []
            glcm_features = [] 
            labels = [] 

            for distance in distances:
              for angle in angles:
         
                glcm = graycomatrix(img, distances=[distance], angles=[angle], symmetric=True, normed=True)
                contrast = graycoprops(glcm, prop='contrast')[0, 0]
                correlation = graycoprops(glcm, prop='correlation')[0, 0]
                energy = graycoprops(glcm, prop='energy')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0][0]

                image_features.extend([contrast, correlation, energy, homogeneity])

  
            glcm_features.append(image_features)

            glcm_features = np.array(glcm_features)
            labels = np.array(labels)

            return glcm_features







