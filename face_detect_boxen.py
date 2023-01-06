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
    
    # Draw a colored box around the face
    if gender == 'male' and age == 'child':
        color = (255, 0, 0) # Blue
    elif gender == 'male' and age == 'adult':
        color = (0, 255, 0) # Green
    elif gender == 'female' and age == 'child':
        color = (0, 0, 255) # Red
    elif gender == 'female' and age == 'adult':
        color = (255, 255, 0) # Yellow
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

# Show the image with the boxes drawn around the faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
