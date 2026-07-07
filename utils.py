import numpy as np


def generate_dataset(n_samples=1000, radius=0.5, random_seed=42):
    """
    Generate a simple dataset of 2D points.

    Each point has two values:
    x1 and x2

    Example:
    [0.2, -0.7]

    The label will be:
    - 1 if the point is inside the circle.
    - 0 if the point is outside the circle.

    Circle logic:
    x1^2 + x2^2 < radius

    NumPy hints:
    Use np.random.seed().
    Use np.random.uniform().
    Use .astype(int).
    Use .reshape(-1, 1).

    Returns:
    X: input points with shape (n_samples, 2)
    y: labels with shape (n_samples, 1)
    """
    # Step 1: Set the random seed so results are repeatable.
    np.random.seed(random_seed)

    # Step 2: Generate random points between -1 and 1.
    # X should have shape (n_samples, 2).
    X = np.random.uniform(-1, 1, size=(n_samples, 2))

    # Step 3: Calculate x1^2 + x2^2 for every point.
    coordinates_sum=np.square(X[:,0])+np.square(X[:,1])

    # Step 4: If x1^2 + x2^2 < radius, label should be 1.
    # Otherwise, label should be 0.
    y=(coordinates_sum<radius).astype(int)

    # Step 5: Reshape y to have shape (n_samples, 1).
    y=y.reshape(-1,1)

    # Step 6: Return X and y.
    return X,y


def train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Split the dataset into training data and testing data.

    Training data:
    Used by the model to learn.

    Testing data:
    Used to check how well the model performs on data it has not seen before.

    Example:
    If we have 1000 samples and test_size = 0.2:
    - 800 samples go to training.
    - 200 samples go to testing.

    NumPy hints:
    Use np.random.seed().
    Use np.random.permutation().
    Use slicing.
    """

    # Step 1: Set the random seed.
    np.random.seed(random_seed)

    # Step 2: Find total number of samples.
    n_samples = X.shape[0]

    # Step 3: Create shuffled indices.
    indices = np.random.permutation(n_samples)

    # Step 4: Decide how many samples should go into test data.
    test_count = int(n_samples * test_size)

    # Step 5: Split indices into test indices and train indices.
    test_indices=indices[:test_count]
    train_indices=indices[test_count:]
    

    # Step 6: Use indices to create:
    # X_train, X_test, y_train, y_test
    X_train=X[train_indices]
    y_train=y[train_indices]
    X_test=X[test_indices]
    y_test=y[test_indices]

    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy.

    Accuracy means:
    How many predictions were correct?

    y_true contains correct labels.
    y_pred contains predicted labels.

    Example:
    Correct labels: [1, 0, 1, 1]
    Predictions:    [1, 0, 0, 1]

    Correct predictions = 3
    Total predictions = 4
    Accuracy = 3 / 4 = 0.75

    NumPy hint:
    Use np.mean().
    """

    accuracy=np.mean((y_true==y_pred).astype(int))
    return  accuracy


# Optional small test.
if __name__ == "__main__":
    X, y = generate_dataset(n_samples=10)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First 5 X values:")
    print(X[:5])
    print("First 5 y values:")
    print(y[:5])