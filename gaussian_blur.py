import cv2 

def get_Gaussian_blur(image):
  img = cv2.resize(image, (256, 400), interpolation = cv2.INTER_AREA)

  blurred = cv2.GaussianBlur(img, (5, 5), 0)


  denoised = cv2.fastNlMeansDenoising(blurred, None, 10, 7, 21)

  return denoised
