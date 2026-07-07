# Tiny Neural Network Using Python and NumPy

## Project Overview

This project is a beginner-friendly implementation of a small neural network using only Python and NumPy.

The model learns to classify 2D points.

Each point has two values:

```text
x1, x2
```

The model predicts whether the point is:

```text
0 = outside the circle
1 = inside the circle
```

## Why This Project Exists

The goal of this project is not to build a large AI system.

The goal is to understand the basic parts of a neural network:

```text
data
weights
biases
activation functions
forward pass
loss
backpropagation
gradient descent
accuracy
```

## Tools Used

```text
Python
NumPy
GitHub
```

## Libraries Not Allowed

```text
TensorFlow
PyTorch
Keras
scikit-learn
pandas
```

## Project Structure

```text
project-root/
|
|-- README.md
|-- requirements.txt
|-- main.py
|-- activations.py
|-- losses.py
|-- neural_network.py
|-- utils.py
|-- report.md
```

## How to Run the Project

Step 1: Clone the repository.

```bash
git clone YOUR_REPOSITORY_URL
```

Step 2: Move into the project folder.

```bash
cd YOUR_PROJECT_FOLDER
```

Step 3: Install requirements.

```bash
pip install -r requirements.txt
```

Step 4: Run the project.

```bash
python main.py
```
We can also choose the hidden activation and save or load weights with command line argument:

```bash
python main.py --activation tanh --epochs 1000 --save-weights weights.npz
python main.py --load-weights weights.npz --activation tanh --epochs 0
```

## Expected Output

Your output may look something like this:

```text
Epoch 0, Loss: 0.693
Epoch 100, Loss: 0.521
Epoch 200, Loss: 0.401
Epoch 300, Loss: 0.320

Final Training Accuracy: 0.86
Final Testing Accuracy: 0.84
```

Your numbers do not have to be exactly the same.

The important thing is:

```text
loss should generally go down
accuracy should generally go up
```

## Activation Functions Implemented

```text
Sigmoid
ReLU
Tanh
Softmax
```

## Final Accuracy

Write your final result here:

```text
Training Accuracy:0.99375
Testing Accuracy:0.98
```

## What I Learned

```text
I revised my neural network concepts and implementation so we have three layers input,hidden and output layer each containg certain ammount of neurons. We do a forward pass in which we multiply the input with weights and pass the result to the next layer, in back propogation we pass the error backwards this way we can readjust our results for a better accuracy. But in university we used to use sklearn and scikit as well for example during splitting data for training and testing but this project required me to use only NumPy which forced me to apply the equations manually giving me a chance to revisit the equations that we only study in theory. I used to use model.fit and this would do the forwards and backward pass and updating parameters pass but using numpy I had to implement this manualy.
```

## Author

```text
Name: Ibrahim Imtiaz
Intern Batch: AI-03
GitHub Profile: Ibrahim-Imtiaz
```
