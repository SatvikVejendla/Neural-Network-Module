# Neural-Network-Module

A simple easy to use Node JS, HTML JS, and Java module which lets you create your own neural network and train it. This was coded from scratch and uses my [numpy-matrix-js](https://github.com/SatvikVejendla/numpy-matrix-js) module for the Matrix math.

- [About](#about)
- [Installation](#set-up)
- [Quick Start (Standard)](#quick-start-standard)
- [Quick Start (DFF)](#quick-start-dff)
- [Examples](#examples)
- [Documentation](#documentation)
- [Versions](#versions)

# About

### Types of Neural Networks supported

- Standard (Feed Forward)
- DFF (Deep Feed Forward) aka multiple hidden layers module

# Set up

- [Browser](https://github.com/SatvikVejendla/Neural-Network-Module/blob/main/src/html/ReadMe.md)
- [NPM](https://github.com/SatvikVejendla/Neural-Network-Module/blob/main/src/npm/ReadMe.md)
- [Java](https://github.com/SatvikVejendla/Neural-Network-Module/blob/main/src/java/ReadMe.md)
- C++ coming soon
# Quick Start Standard

### Creating a neural network

The function to create a new neural network is just:

```
const nn = new NeuralNetwork.Standard(input_nodes, hidden_nodes, output_nodes);
```

In this case, you will have to change some of these.

1. replace the `input_nodes` with however many inputs you give the network
2. replace the `hidden_nodes` with however many hidden nodes you want. this number is arbitrary. You can choose whatever you want but might have to tweak it for optimal results.
3. replace the `output_nodes` with however many outputs you want.

Once you're done with this, you have finished creating the neural network. The next step is to train it with data.

### Training the model

To train the model, use the following function:

```
nn.train(input, output)
```

where `input` is the value that you input and `output` is the value the computer should output.

### Testing the model

To test the model, use this function:

```
nn.predict(input)
```

This will make the computer use it's previous tested data to make a guess for what the output should be.

You're done creating a basic neural network. For more functionality, take a look at the documentation.

# Quick Start DFF

### Creating a neural network

The function to create a new neural network is just:

```
const nn = new NeuralNetwork.DFF(input_nodes, hidden_nodes, output_nodes);
```

In this case, you will have to change some of these.

1. replace the `input_nodes` with however many inputs you give the network
2. replace the `hidden_nodes` with an array. The first element in the array is the number of hidden nodes for the first hidden layer, the second element is for the second layer, and so on.
3. replace the `output_nodes` with however many outputs you want.

Once you're done with this, you have finished creating the neural network. The next step is to train it with data.

### Training the model

To train the model, use the following function:

```
nn.train(input, output)
```

where `input` is the value that you input and `output` is the value the computer should output.

### Testing the model

To test the model, use this function:

```
nn.predict(input)
```

Finished.

# Examples

- Standard
  - [XOR Example](https://github.com/SatvikVejendla/Neural-Network-Node/blob/main/examples/Standard/XOR.js)
  - [Sine Graph](https://github.com/SatvikVejendla/Neural-Network-Node/blob/main/examples/Standard/sinwave.js)
- DFF
  - [Sine Graph](https://github.com/SatvikVejendla/Neural-Network-Node/blob/main/examples/DFF/sinwave.js)

# Documentation

`nn.getWeights()`

Returns all of the current weights of the neural network

`nn.getBias()`

Returns all of the current biases of the neural network

`nn.getLearningRate()`

Returns the current learning rate of the neural network

`nn.setWeights()`

| Parameters    | What it is       | Required |
| ------------- | ---------------- | -------- |
| weights_array | Array of weights | Yes      |

`nn.setBias()`

| Parameters | What it is      | Required |
| ---------- | --------------- | -------- |
| bias_array | Array of biases | Yes      |

`nn.setLearningRate()`

| Parameters    | What it is        | Required |
| ------------- | ----------------- | -------- |
| learning_rate | New learning rate | Yes      |

`nn.predict()`

| Parameters  | What it is                                                 | Required |
| ----------- | ---------------------------------------------------------- | -------- |
| input_array | Array of input data that matches the number of input_nodes | Yes      |

`nn.train()`

| Parameters   | What it is                      | Required |
| ------------ | ------------------------------- | -------- |
| input_array  | the input data                  | Yes      |
| output_array | what the computer should output | Yes      |

# Versions

Current Version: 1.3.9

Stable Versions:

**1.3.9** - Changed Matrix library to my [numpy-matrix-js](https://github.com/SatvikVejendla/numpy-matrix-js) module for NPM versions only.

**1.3.1** - Added DFF for browser support

**1.2.9** - Added browser support

**1.2.1** - Succesfully added Deep Feed Forward Network. Removed error message for mismatched input data client-side.

**1.1.7** - Failed to add Perceptron support. Forced to remove it temporarily

**1.1.0** - First stable version
