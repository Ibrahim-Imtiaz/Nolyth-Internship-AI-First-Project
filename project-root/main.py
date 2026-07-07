import argparse

from utils import generate_dataset, train_test_split, accuracy_score
from losses import binary_cross_entropy
from neural_network import (
    initialize_parameters,
    forward_pass,
    backward_pass,
    update_parameters,
    predict,
    save_parameters,
    load_parameters,
)

#I made this to parse the Commands pass thriugh Terminal
def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny neural network with NumPy.")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--load-weights", type=str, default=None)
    parser.add_argument("--save-weights", type=str, default=None)
    return parser.parse_args()


def train_model(args):
    """
    Train the neural network from start to finish.

    Full training flow:

    1. Create dataset.
    2. Split dataset into training and testing parts.
    3. Initialize weights and biases.
    4. Repeat training for many epochs.
    5. Print loss after some epochs.
    6. Calculate final accuracy.
    """

    # -----------------------------
    # Step 1: Project settings
    # -----------------------------

    # You can change these values later and see what happens.
    n_samples = args.n_samples
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    activation_function = args.activation

    # Fixed sizes for this project.
    input_size = 2
    output_size = 1

    # -----------------------------
    # Step 2: Generate dataset
    # -----------------------------
    X,y=generate_dataset(n_samples=n_samples)

    # -----------------------------
    # Step 3: Split dataset
    # -----------------------------
    X_train, X_test, y_train, y_test=train_test_split(X,y)

    # -----------------------------
    # Step 4: Initialize parameters
    # -----------------------------
    if args.load_weights:
        parameters = load_parameters(args.load_weights)
    else:
        parameters = initialize_parameters(input_size, hidden_size, output_size)

    # -----------------------------
    # Step 5: Training loop
    # -----------------------------

    for i in range(epochs):
        y_pred,cache=forward_pass(X_train,parameters,activation_function)
        loss=binary_cross_entropy(y_train,y_pred)
        gradient=backward_pass(X_train, y_train, parameters, cache,activation_function)
        parameters=update_parameters(parameters,gradient,learning_rate)

        if i%100==0:
            print("epoch no: ",i,", loss:",loss)

    # -----------------------------
    # Step 6: Final predictions
    # -----------------------------
    training_predicitons=predict(X_train,parameters,activation_function)
    testing_predicitons=predict(X_test,parameters,activation_function)

    # -----------------------------
    # Step 7: Calculate accuracy
    # -----------------------------
    training_accuracy=accuracy_score(y_train,training_predicitons)
    testing_accuracy=accuracy_score(y_test,testing_predicitons)

    # -----------------------------
    # Step 8: Print final result
    # -----------------------------

    print("Final Training Accuracy:",training_accuracy)
    print("Final Testing Accuracy:",testing_accuracy)

    if args.save_weights:
        save_parameters(parameters, args.save_weights)
        print("Saved weights to:", args.save_weights)


if __name__ == "__main__":
    train_model(parse_args())