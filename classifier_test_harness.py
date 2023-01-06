'''
'''

import cv2
import numpy as np
from sklearn.externals import joblib

# Load the classifier
classifier = joblib.load('nudity_classifier.pkl')

# Load the test dataset of images
X_test = []
y_test = []
for i in range(1, 101):
    # Load the image
    image = cv2.imread("test_set/{}.jpg".format(i))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract features from the image using histogram of oriented gradients (HOG)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Add the features to the test dataset
    X_test.append(features)
    
    # Set the label for the image
    if "nudity" in "test_set/{}.jpg".format(i):
        y_test.append(1)
    else:
        y_test.append(0)

# Test the classifier on the test dataset
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)
