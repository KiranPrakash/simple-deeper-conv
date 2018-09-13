from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2


# Utility function to convert the input raw image in any resolution to model input size resolution
def image_to_feature_vector(image, size=(28, 28)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, size).flatten()


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help=' path to the output model file')
ap.add_argument('-t', '--test-images', required=True, help='path to the test images')
ap.add_argument('-b', '--batch-size', type=int, default=32, help='size of the mini batch size')
args = vars(ap.parse_args())

# initialize the class labels for the kaggle dog vs cats dataset
model = load_model(args["model"])

# loop over out testing images
for imagePath in paths.list_images(args["test_images"]):
    # load the image, resize it to a fixed pixels ignoring the aspect ratio and then extract from it
    image = cv2.imread(imagePath)
    image_float = image.astype('float32')
    features = image_to_feature_vector(image_float) / 255.0
    features = np.array([features])

    #Note: input_shape to the predict should be of (img_rows, img_cols, 1)
    features = features.reshape(1, 28, 28, 1)
    # classify the image using our extracted features and pre-trained neural network
    probability = model.predict(features)[0]
    prediction = probability.argmax(axis=0)
    print("Predicted digit: {}, Probability: {:.3f}".format(prediction, max(probability)))

    # draw the class and probability on the test image and display to our screen
    label = "Digit {}, Prob= {:.2f}".format(prediction, max(probability))
    image = cv2.resize(image, (280,280), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
