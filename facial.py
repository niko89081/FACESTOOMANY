import cv2 as cv
from deepface import DeepFace
cascPath = "something.xml"
faceCascade = cv.CascadeClassifier(cascPath)
image = cv.imread("Screenshot 2021-02-07 023640.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags= cv.CASCADE_SCALE_IMAGE
)
print(faces)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
obj = DeepFace.analyze(img_path = "Screenshot 2021-02-07 023640.png", actions= ['age', 'gender', 'race', 'emotion'])
print(obj)
cv.imshow("Faces found", image)
cv.waitKey(0)
cv.destroyAllWindows()