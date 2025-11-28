# unzip the dataset
import zipfile
with zipfile.ZipFile("/content/PetImages.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/")

# importing required libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# setting the dataset path
data_path = "/content/PetImages"
cats_folder = data_path + "/cats"   # path to cat images
dogs_folder = data_path + "/dogs"   # path to dog images

# lists to store data and tables
X = []    # will store image data
y = []    # will store labels (0 = cat, 1 = dog)
img_size = 64   # resize all images to 64x64 pixels

# loading the cat images
for img_name in os.listdir(cats_folder):
    try:
        img = cv2.imread(os.path.join(cats_folder, img_name))   # read the image
        img = cv2.resize(img, (img_size, img_size))             # resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # convert to grayscale
        X.append(img.flatten())                                 # flatten image to 1D array
        y.append(0)   # label 0 for cats
    except:
        pass   # ignore unreadable images

# loading the dog images
for img_name in os.listdir(dogs_folder):
    try:
        img = cv2.imread(os.path.join(dogs_folder, img_name))   # read image
        img = cv2.resize(img, (img_size, img_size))             # resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # grayscale
        X.append(img.flatten())                                 # flatten
        y.append(1)   # label 1 for dogs
    except:
        pass   # ignore unreadable images

# convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# training the svm model
model = SVC(kernel='linear')      # using linear SVM
model.fit(X_train, y_train)       # train the model

# making predictions
y_pred = model.predict(X_test)

# calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# testing the model with a single image
import cv2
import numpy as np
def predict_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Image not found -> {img_path}")
            return
        img = cv2.resize(img, (img_size, img_size))    # resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # grayscale
        img = img.flatten()                            # flatten
        img = img.reshape(1, -1)                       # reshape for model
        prediction = model.predict(img)[0]

        if prediction == 0:
            print(f"{img_path} → 🐱 CAT")
        else:
            print(f"{img_path} → 🐶 DOG")

    except Exception as e:
        print("Error:", e)

# NOW TEST BOTH IMAGES
predict_image("/content/cat.1.jpg")
predict_image("/content/dog.4001.jpg")
