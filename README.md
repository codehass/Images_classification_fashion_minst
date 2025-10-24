### Fashion MNIST Image Classification

This project implements image classification on the Fashion MNIST dataset using two different neural network architectures in Python with TensorFlow and Keras.

## The notebook explores:

- A Multilayer Perceptron (MLP) architecture for image classification.
- A Convolutional Neural Network (CNN) architecture for image classification.

## Dataset

The Fashion MNIST dataset contains 60,000 28x28 grayscale images of 10 fashion categories (e.g., T-shirts, dresses, shoes) for training, and 10,000 images for testing.

- Training Data: 60,000 images.

- Testing Data: 10,000 images.

- Categories: 0. T-shirt/top

  1. Trouser
  2. Pullover
  3. Dress
  4. Coat
  5. Sandal
  6. Shirt
  7. Sneaker
  8. Bag
  9. Ankle boot

  ## Requirements

  To run this project, you will need the following libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas

You can install the required libraries with the following commands:

```shell
  pip install tensorflow numpy matplotlib pandas
```

## Project Structure

1- Data Preprocessing: The dataset is loaded and normalized.
2- Model Architectures:

- MLP (Multilayer Perceptron): A fully connected feedforward neural network with two hidden layers.
- CNN (Convolutional Neural Network): A CNN with convolutional and pooling layers followed by dense layers.
  3- Training: The models are trained using the Fashion MNIST dataset.
  4- Evaluation: The models are evaluated on test data, and accuracy is reported.

## Architecture

1. MLP Architecture

```python
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

## The MLP model consists of:

- A Flatten layer to reshape the 28x28 input images into a 1D vector.
- Two Dense layers with ReLU activations for learning features.
- A final Dense output layer with 10 neurons and a softmax activation to classify into 10 categories.

2. CNN Architecture

```python
modelCNN = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

```

The CNN model consists of:

- Conv2D layers for feature extraction from images.
- MaxPooling2D layers for dimensionality reduction.
- A Flatten layer followed by Dense layers for classification.

## Training & Evaluation

- The models are compiled with the Adam optimizer and sparse categorical crossentropy loss function.
- The models are trained for 10 epochs.
- The performance of each model is evaluated on the test dataset, and the test accuracy is reported.

## Results

- MLP model accuracy: (0.878)
- CNN model accuracy: (0.908)

## Conclusion

This project demonstrates the effectiveness of deep learning architectures (MLP vs. CNN) in classifying fashion items from the Fashion MNIST dataset. CNNs tend to perform better on image classification tasks due to their ability to learn spatial hierarchies of features.
