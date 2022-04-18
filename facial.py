import cv2 as cv
from deepface import DeepFace
cascPath = "something.xml"
faceCascade = cv.CascadeClassifier(cascPath)
vid = cv.VideoCapture(0)
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if len(faces):
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    cv.imshow('frame', frame)
# After the loop release the cap object
# Destroy all the windows
cv.destroyAllWindows()

print(faces)

# Draw a rectangle around the faces
cv.waitKey(0)
cv.destroyAllWindows()
