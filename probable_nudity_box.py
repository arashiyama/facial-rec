'''

'''
import cv2
import numpy as np
from sklearn.externals import joblib

# Load the classifier
classifier = joblib.load('nudity_classifier.pkl')

# Load the picture
image = cv2.imread("picture.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract features from the image using histogram of oriented gradients (HOG)
features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

# Classify the image as containing probable nudity or not
probability = classifier.predict_proba([features])[0][1]
if probability > 0.5:
    print("Probable nudity detected")
    # Find the bounding box of the probable nudity
    x, y, w, h = classifier.predict(features)
    # Draw a box around the probable nudity
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
else:
    print("No probable nudity detected")

# Show the image with the box drawn around the probable nudity
cv2.imshow("Probable Nudity", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
