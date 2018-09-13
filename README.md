# simple-deeper-conv : MNIST Digit Classifier 
MNIST is one of basic computer vision datasets, which consists of images of handwritten
digits.

# Description:
The repo contains two python executables.
1. simple_deep_conv.py:mplements the neural network architecture and training pipeline 
2. infer_model.py: Implements a simple inference server with the previously trained model in step1 with a raw input image MNIST digit tests sets


# Training with MNIST dataset directly loaded from Tensorflow Keras API
1. Create a output model file name which you may want to store the model file after training in the project root directory
2. And from the root directory simply execute the python file. This will create the trained model in *.hdf5 format

```sh
$ python3 simple_deep_conv.py
```

# Inferencing on Raw Input Image:
1. Use the command line to execute the infer_model.py script
2. This could be executed from the project root directory as 
```sh
$ python3 infer_model.py -m simple_deep_conv.py -t test_images/image.png
```
 
 
# External Libraries Used:
 1. OpenCV2
 2. TensorFlow Keras API
 3. Imutils (utility tool)

