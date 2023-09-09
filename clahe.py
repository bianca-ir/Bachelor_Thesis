import cv2 

def get_CLAHE(image):
  image = cv2.resize(image, (256, 400), interpolation = cv2.INTER_AREA)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

  clahe_image = clahe.apply(image)

  return clahe_image