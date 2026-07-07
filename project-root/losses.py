import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """
    Binary cross-entropy is used when we have only two classes.

    In this project:
    - 0 means outside the circle.
    - 1 means inside the circle.

    y_true contains the correct answers.
    y_pred contains the model predictions.

    Formula:
    Loss = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    Why do we use np.clip()?
    Sometimes y_pred can become exactly 0 or exactly 1.
    log(0) is not safe and can break the program.
    np.clip() keeps values in a safe range.

    NumPy hints:
    Use np.clip().
    Use np.log().
    Use np.mean().
    """
    # Step 1: Clip y_pred so values stay between 1e-8 and 1 - 1e-8.
    # Step 2: Write the binary cross-entropy formula.
    # Step 3: Return the loss.
    y_pred=np.clip(y_pred,1e-8,1-1e-8)
    loss=-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss



def mean_squared_error(y_true, y_pred):
    """
    Optional bonus function.

    Mean Squared Error checks the average squared difference between
    correct answers and predictions.

    This is not required for the main project.
    It is only for extra practice.

    Formula:
    MSE = mean((y_true - y_pred)^2)

    NumPy hint:
    Use np.mean().
    """
    # Step 1: Subtract y_pred from y_true.
    # Step 2: Square the result.
    # Step 3: Take the mean.
    # Step 4: Return the result.
    return np.mean(np.square(y_true-y_pred))


if __name__ == "__main__":
    y_true = np.array([[1], [0], [1], [0]])
    y_pred = np.array([[0.9], [0.2], [0.8], [0.1]])
    print("Binary Cross Entropy:", binary_cross_entropy(y_true, y_pred))