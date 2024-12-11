"""import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path="test"

def getImgID(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        faceImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        faceNp=np.array(faceImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        Ids.append(Id)
        #cv2.imshow("training",faceNp)
        #cv2.waitKey(10)
        # extract the face from the training image sample
        #faces=detector.detectMultiScale(imageNp)
         #If a face is there then append that in the list as well as Id of it
    return Ids,faces

    
Ids,faces = getImgID(path)
#print(Ids,faces)
recognizer.train(faces, np.array(Ids))
recognizer.write('recognizer\\training_data.yml')
cv2.destroyAllWindows()
path = 'path_to_your_dataset_directory'  # Update this line
path = r"C:\crime\CFIS-criminal-face-identification-system-master\test"""

import os

def getImgID(path):
    Ids = []
    faces = []
    for imagePath in os.listdir(path):
        # Check if the file is an image file
        if imagePath.endswith(('.jpg', '.jpeg', '.png')):  # Add other extensions if necessary
            # Split the filename and extract the ID
            parts = os.path.split(imagePath)[-1].split(".")
            if len(parts) > 1:  # Ensure there is a part after the first dot
                try:
                    Id = int(parts[1])  # Try to convert the second part to an integer
                    Ids.append(Id)
                    # Assuming you have some way to get the face data
                    # faces.append(get_face_data(imagePath))  # Replace with actual face extraction logic
                except ValueError:
                    print(f"Warning: Could not convert '{parts[1]}' to an integer in file '{imagePath}'")
            else:
                print(f"Warning: Filename '{imagePath}' does not contain an ID.")
    return Ids, faces

# Example usage
path = 'path_to_your_images'  # Replace with your actual path
Ids, faces = getImgID(path)