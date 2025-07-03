# Reference

https://www.youtube.com/watch?v=d2kxUVwWWwU&t=4354s&pp=ygUZRGVlcCBsZXJhcm5pbmgga3Jpc2ggbmFpaw%3D%3D 


# Documentation Comment

This code provides an introduction to the basics of deep learning. It demonstrates fundamental concepts such as defining neural network architectures, initializing model parameters, and implementing forward and backward propagation. The code is structured to help beginners understand how deep learning models are built and trained.

## Key Features:
1. **Neural Network Architecture**: Defines the layers, activation functions, and connections between neurons.
2. **Parameter Initialization**: Sets up weights and biases for the model.
3. **Forward Propagation**: Computes the output of the network by passing input data through the layers.
4. **Backward Propagation**: Calculates gradients to update model parameters using optimization techniques.
5. **Training Loop**: Iteratively adjusts parameters to minimize the loss function.

## Usage:
- This code is intended for educational purposes and can be extended for more complex deep learning tasks.
- Ensure that required libraries (e.g., NumPy, TensorFlow, or PyTorch) are installed before running the code.

## Notes:
- The code assumes familiarity with basic programming concepts and linear algebra.
- For practical applications, consider using established deep learning frameworks for efficiency and scalability.

## Forward Propagation Explained

Forward propagation is the process of passing input data through a neural network to compute the output. It involves a series of mathematical operations performed layer by layer. Here's a step-by-step explanation:

1. **Input Layer**:
    - The input data is fed into the network. Each feature of the input corresponds to a neuron in the input layer.

2. **Weighted Sum**:
    - For each neuron in the next layer, compute the weighted sum of inputs:
      \[
      z = \sum (w \cdot x) + b
      \]
      where:
      - \( w \): weight of the connection
      - \( x \): input value
      - \( b \): bias term

3. **Activation Function**:
    - Apply an activation function to the weighted sum to introduce non-linearity:
      \[
      a = f(z)
      \]
      Common activation functions include ReLU, Sigmoid, and Tanh.

4. **Repeat for Hidden Layers**:
    - The output of one layer becomes the input for the next layer. Repeat the weighted sum and activation steps for all hidden layers.

5. **Output Layer**:
    - The final layer produces the network's output. For classification tasks, this might involve a softmax function to generate probabilities.

### Example:
For a single neuron:
\[
z = w_1x_1 + w_2x_2 + b
\]
\[
a = \text{ReLU}(z)
\]

Forward propagation is computationally efficient and forms the basis for making predictions in a neural network.
