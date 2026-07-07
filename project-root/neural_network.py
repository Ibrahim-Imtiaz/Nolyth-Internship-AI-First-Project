import numpy as np
from activations import sigmoid, relu, relu_derivative,sigmoid_derivative,tanh_derivative


def initialize_parameters(input_size, hidden_size, output_size, random_seed=42):
    """
    Create the starting weights and biases.

    What are weights?
    Weights are numbers that the model changes during training.

    What are biases?
    Biases are extra numbers that help the model adjust its output.

    For this project:
    input_size = 2
    hidden_size = 4 to 8
    output_size = 1

    Required shapes:
    W1: (input_size, hidden_size)
    b1: (1, hidden_size)

    W2: (hidden_size, output_size)
    b2: (1, output_size)

    NumPy hints:
    Use np.random.seed().
    Use np.random.randn().
    Use np.zeros().

    Returns:
    parameters dictionary containing W1, b1, W2, b2
    """
    # Step 1: Set the random seed.
    np.random.seed(random_seed)

    # Step 2: Create W1 with small random values.
    W1=np.random.randn(input_size, hidden_size)*0.01

    # Step 3: Create b1 with zeros.
    b1=np.zeros((1, hidden_size))

    # Step 4: Create W2 with small random values.
    W2=np.random.randn(hidden_size,output_size)*0.01

    # Step 5: Create b2 with zeros.
    b2=np.zeros((1, output_size))

    # Step 6: Store all values in a dictionary.
    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    # Step 7: Return parameters.
    return parameters


def save_parameters(parameters, file_path):
    """Save model weights and biases to a NumPy .npz file."""
    np.savez(file_path, **parameters)
def load_parameters(file_path):
    """Load model weights and biases from a NumPy .npz file."""
    data = np.load(file_path)
    parameters = {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"]
    }
    print("Model loaded from",file_path)
    return parameters

def forward_pass(X, parameters,activation_function="relu"):
    """
    Move data through the neural network.

    This is called the forward pass.

    Simple meaning:
    The model receives input X and gives an output prediction.

    Formula:

    Layer 1:
    Z1 = X @ W1 + b1
    A1 = ReLU(Z1)

    Layer 2:
    Z2 = A1 @ W2 + b2
    A2 = Sigmoid(Z2)

    A2 is the final prediction.

    Shape guide:
    X:  (m, 2)
    W1: (2, hidden_size)
    b1: (1, hidden_size)

    Z1: (m, hidden_size)
    A1: (m, hidden_size)

    W2: (hidden_size, 1)
    b2: (1, 1)

    Z2: (m, 1)
    A2: (m, 1)

    NumPy hints:
    Use @ for matrix multiplication.
    Use relu().
    Use sigmoid().

    Returns:
    A2: final prediction
    cache: stored values needed later for backpropagation
    """
    # Step 1: Get W1, b1, W2, b2 from the parameters dictionary.
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    
    # Step 2: Calculate Z1.
    Z1=X@W1+b1

    # Step 3: Apply ReLU to Z1.
    if activation_function=="relu":
        A1=relu(Z1)
    elif activation_function=="tanh":
        A1=np.tanh(Z1)
    elif activation_function=="sigmoid":
        A1=sigmoid(Z1)
    else:
        print("activation function must be relu,tanh,sigmoid")

    # Step 4: Calculate Z2.
    Z2=A1@W2+b2

    # Step 5: Apply sigmoid to Z2.
    A2=sigmoid(Z2)

    # Step 6: Store Z1, A1, Z2, A2 in a cache dictionary.
    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }

    # Step 7: Return A2 and cache.
    return A2,cache


def backward_pass(X, y, parameters, cache,activation_function="relu"):
    """
    Calculate gradients.

    This is called backpropagation.

    Do not be scared by this word.

    Simple meaning:
    Backpropagation tells us how to change weights and biases so the model
    can make better predictions next time.

    You do not need to derive these formulas yourself.
    Your job is to carefully convert them into NumPy code.

    Formulas:

    m = number of training examples

    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / m
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    NumPy hints:
    Use .T for transpose.
    Use @ for matrix multiplication.
    Use np.mean(..., axis=0, keepdims=True).

    Returns:
    gradients dictionary containing dW1, db1, dW2, db2
    """

    # Step 1: Find the number of examples.
    m=X.shape[0]

    # Step 2: Get W2 from parameters.
    W2=parameters["W2"]

    # Step 3: Get Z1, A1, A2 from cache.
    Z1=cache["Z1"]
    A1=cache["A1"]
    A2=cache["A2"]

    
    # Step 4: Calculate dZ2.
    dZ2=A2-y

    # Step 5: Calculate dW2.
    dW2=A1.T@dZ2/m

    # Step 6: Calculate db2.
    db2=np.mean(dZ2,axis=0,keepdims=True)

    # Step 7: Calculate dA1.
    dA1=dZ2@W2.T

    # Step 8: Calculate dZ1.
    if activation_function=="relu":
        dZ1=dA1*relu_derivative(Z1)
    elif activation_function=="tanh":
        dZ1=dA1*tanh_derivative(Z1)
    elif activation_function=="sigmoid":
        dZ1=dA1*sigmoid_derivative(Z1)


    # Step 9: Calculate dW1.
    dW1=X.T@dZ1/m

    # Step 10: Calculate db1.
    db1=np.mean(dZ1, axis=0, keepdims=True)

    # Step 11: Store gradients in a dictionary.

    gradients={
         "dW1": dW1,
         "db1": db1,
         "dW2": dW2,
         "db2": db2,
    }

    # Step 12: Return gradients.
    return gradients


def update_parameters(parameters, gradients, learning_rate):
    """
    Update weights and biases.

    Simple meaning:
    The model has made a mistake.
    Gradients tell us the direction of improvement.
    Learning rate controls how big the update step should be.

    Formula:

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    Returns:
    updated parameters dictionary
    """

    # Step 1: Get W1, b1, W2, b2 from parameters.
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    # Step 2: Get dW1, db1, dW2, db2 from gradients.
    dW1=gradients["dW1"]
    db1=gradients["db1"]
    dW2=gradients["dW2"]
    db2=gradients["db2"]

    # Step 3: Update W1.
    W1=W1-learning_rate*dW1
    
    # Step 4: Update b1.
    b1=b1-learning_rate*db1
    
    # Step 5: Update W2.
    W2=W2-learning_rate*dW2
    
    # Step 6: Update b2.
    b2=b2-learning_rate*db2

    # Step 7: Store updated values back into parameters dictionary.
    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    # Step 8: Return parameters.
    return parameters


def predict(X, parameters, activation_function="relu"):
    """
    Make final predictions.

    The model output from sigmoid is a number between 0 and 1.

    Example:
    0.91 means the model is confident the answer is 1.
    0.18 means the model is confident the answer is 0.

    Rule:
    If output >= 0.5, predict 1.
    If output < 0.5, predict 0.

    Returns:
    predictions with shape (m, 1)
    """

    # Step 1: Run forward_pass() to get probabilities.
    probabilities,_=forward_pass(X,parameters,activation_function)

    # Step 2: Convert probabilities into 0 or 1.
    predictions=(probabilities>=0.5).astype(int)

    # Step 3: Return predictions.
    return predictions