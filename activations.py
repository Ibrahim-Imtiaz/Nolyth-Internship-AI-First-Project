import numpy as np


def sigmoid(x):
    """
    Sigmoid changes any number into a value between 0 and 1.

    Example:
    - A big positive number becomes close to 1.
    - A big negative number becomes close to 0.
    - 0 becomes 0.5.

    Why do we use it?
    In this project, the final answer should behave like a probability.
    That is why sigmoid is used in the output layer.

    Formula:
    sigmoid(x) = 1 / (1 + e^(-x))

    NumPy hint:
    Use np.exp().
    """
    # Step 1: Write the sigmoid formula using np.exp().
    # Step 2: Return the result.
    sigmoid_result= 1/(1+np.exp(-x))
    return sigmoid_result


def relu(x):
    """
    ReLU means Rectified Linear Unit.

    Simple meaning:
    - If the number is positive, keep it.
    - If the number is negative, turn it into 0.

    Example:
    - relu(-5) = 0
    - relu(3) = 3

    Why do we use it?
    ReLU helps the neural network learn patterns that are not just straight lines.

    NumPy hint:
    Use np.maximum().
    """
    # Step 1: Compare x with 0.
    # Step 2: Return the maximum value.
    return np.maximum(0,x)


def tanh(x):
    """
    Tanh changes numbers into values between -1 and 1.

    Example:
    - A big positive number becomes close to 1.
    - A big negative number becomes close to -1.
    - 0 becomes 0.

    NumPy hint:
    Use np.tanh().
    """
    # Step 1: Use np.tanh(x).
    # Step 2: Return the result.
    return np.tanh(x)


def softmax(x):
    """
    Softmax is usually used when we have more than two classes.

    Example:
    If the classes are cat, dog, and car, softmax can convert raw scores
    into probabilities for each class.

    In this project, we only have two classes:
    - 0 means outside the circle.
    - 1 means inside the circle.

    So softmax is not required in the final model, but you will still
    implement it for practice.

    NumPy hints:
    Use np.exp().
    Use np.sum(..., axis=1, keepdims=True).

    Important:
    Subtracting np.max(x) before np.exp(x) helps avoid very large numbers.
    """
    # Step 1: Make the values stable by subtracting the max value from x.
    # Step 2: Apply np.exp().
    # Step 3: Divide by the sum of exponentials.
    # Step 4: Return the result.
    exp_x=np.exp(x-np.max(x))
    return exp_x/np.sum(exp_x,axis=1,keepdims=True)



def sigmoid_derivative(x):
    """
    Derivative of sigmoid.

    Simple meaning:
    This helps during backpropagation.
    It tells us how sigmoid changes when the input changes.

    Hint:
    First calculate sigmoid(x).
    Then use:
    sigmoid(x) * (1 - sigmoid(x))
    """
    # Step 1: Calculate s = sigmoid(x).
    # Step 2: Return s * (1 - s).
    return sigmoid(x)*(1-sigmoid(x))


def relu_derivative(x):
    """
    Derivative of ReLU.

    Simple meaning:
    - If x is greater than 0, return 1.
    - If x is less than or equal to 0, return 0.

    This helps during backpropagation.

    NumPy hint:
    Use x > 0.
    Use .astype(float).
    """
    # Step 1: Check where x is greater than 0.
    # Step 2: Convert True/False values into 1.0/0.0.
    # Step 3: Return the result.
    return (x > 0).astype(int)

def tanh_derivative(x):
    """
    Derivative of tanh.

    Hint:
    tanh_derivative(x) = 1 - tanh(x)^2
    """
    # Step 1: Calculate np.tanh(x).
    # Step 2: Square it.
    # Step 3: Subtract from 1.
    # Step 4: Return the result.
    return 1-np.square(np.tanh(x))

if __name__ == "__main__":
    x = np.array([-2, -1, 0, 1, 2])

    print("Input:", x)
    print("Sigmoid:", sigmoid(x))
    print("ReLU:", relu(x))
    print("Tanh:", tanh(x))
    print("Softmax:", softmax(x.reshape(1, -1)))
