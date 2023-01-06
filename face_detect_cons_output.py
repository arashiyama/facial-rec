'''
This script uses the OpenCV library to detect and classify faces in an image. 
It first uses a Haar cascade classifier trained for face detection to detect faces in the image. 
It then uses separate Haar cascade classifiers trained for gender and age classification to classify the detected faces as male or female, and as children or adults. 
Note that the Haar cascades for gender and age classification are not included in the OpenCV library and must be trained separately. 
You can find more information on how to do this in the OpenCV documentation.
'''
import cv2

# Load the picture
image = cv2.imread("picture.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar cascades for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Iterate over the detected faces
for (x, y, w, h) in faces:
    # Crop the face from the image
    face = image[y:y+h, x:x+w]
    
    # Load the pre-trained Haar cascades for gender classification
    gender_cascade = cv2.CascadeClassifier('haarcascade_gender.xml')
    
    # Classify the face as male or female
    gender = gender_cascade.predict(face)
    
    # Load the pre-trained Haar cascades for age classification
    age_cascade = cv2.CascadeClassifier('haarcascade_age.xml')
    
    # Classify the face as a child or adult
    age = age_cascade.predict(face)
    
    # Print the results
    if gender == 'male' and age == 'child':
        print("Male child face detected")
    elif gender == 'male' and age == 'adult':
        print("Male adult face detected")
    elif gender == 'female' and age == 'child':
        print("Female child face detected")
    elif gender == 'female' and age == 'adult':
        print("Female adult face detected")
