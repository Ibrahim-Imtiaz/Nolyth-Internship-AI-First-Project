# Project Report: Tiny Neural Network Using NumPy

## 1. What is a Neural Network?

```text
 A neural network basically ressemble the neuron of a human brain, a neuron receives an input and later processes implictly and gives signals as output. In the same manner a neural network is made up of multiple input, hidden and output neurons. The input values from input layer are used by neurons at hidden layer, who multiplies those input values with weights and then ater add a bias, later passing that sum to next layer, the same process of weights multiplication and addition of bias happens unti we reach output layer. We aso use activittion function on the resut of the previous layer. Activitation function invloves functions such as Sigmoid,ReLU,LeakyReLU,tah etc.
```

## 2. What is a Feature?

```text
Features in simple words are number of parameters of input that we are considering. For example in this case we were considering two parametrs of input coordinate, that were x1 and x2. If for example we are making a facial recognization neural network the input features might be forhead width,eyes width,eye shape etc.
```

## 3. What is a Label?

```text
Label is the final verdict of output that we received. For example if we are making cat vs dog recognization system then labels are Cat and Dog. Labels are not bounded to be ony two they can be multiple.
```

## 4. What is an Activation Function?

```text
Activitation function heps in learning phase of model by enabling the model to learn complex patterns instead of just linear patterns.
```

## 5. What is Loss?

```text
Loss is the difference between the predicted value and target value.
```

## 6. What is Gradient Descent?

```text
Through gradient descent we adjust our weights and biases. It is like us coming down from a hill, if the slope is steep we tilt our foot more, else if the surface is more flat we tilt our foot upwards. Similalry while training we have to adjust our gradient according to the weight changes.
```

## 7. What is Backpropagation?

```text
Through back propogation we transfer the error backwards in th chain of neuron, during this phase we are adjusting our weihts to minimize loss/error.
```

## 8. What Was the Most Difficult Part?

```text
Back propogation is usually considered to be most difficult part and similarly I also felt the implementation of this part difficult as there are alot of equations involved, but since I have already implemented these things in my University so it was easier than the first time.
```

## 9. How Did You Solve the Difficulty?

```text
I used the equations provided in the resources. And followed the steps in comments.
```

## 10. What Did You Learn From This Project?

```text
I revised my neural network concepts and implementation so we have three layers input,hidden and output layer each containg certain ammount of neurons. We do a forward pass in which we multiply the input with weights and pass the result to the next layer, in back propogation we pass the error backwards this way we can readjust our results for a better accuracy. But in university we used to use sklearn and scikit as well for example during splitting data for training and testing but this project required me to use only NumPy which forced me to apply the equations manually giving me a chance to revisit the equations that we only study in theory. I used to use model.fit and this would do the forwards and backward pass and updating parameters pass but using numpy I had to implement this manualy.
```