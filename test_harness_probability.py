'''
This script loads the trained classifier and a test dataset of images, and uses the histogram of oriented gradients (HOG) 
feature extraction method to extract features from each image in the test dataset. It then uses the classifier to predict the probability that each image contains 
probable nudity, and outputs the results.

This test harness can be used to evaluate the performance of the nudity probability detector on the test dataset, 
and to identify any errors or false positives in the predictions. You can modify the script to suit your specific needs, 
such as by changing the threshold probability for classifying an image as containing probable nudity, or by adding additional evaluation metrics or visualizations.
'''
import cv2
import numpy as np
from sklearn.externals import joblib

# Load the classifier
classifier = joblib.load('nudity_classifier.pkl')

# Load the test dataset of images
for i in range(1, 101):
    # Load the image
    image = cv2.imread("test_set/{}.jpg".format(i))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract features from the image using histogram of oriented gradients (HOG)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Classify the image as containing probable nudity or not
    probability = classifier.predict_proba([features])[0][1]
    if probability > 0.5:
        print("Image {}: Probable nudity detected (probability = {:.2f})".format(i, probability))
    else:
        print("Image {}: No probable nudity detected (probability = {:.2f})".format(i, probability))
