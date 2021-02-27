# Neural-Network-Module
A simple easy to use Node JS, HTML JS, and Java module which lets you create your own neural network and train it. This was coded from scratch and has no external module dependencies.


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


### NPM
Begin by installing the node module with this command:

```npm install neural-network-node```

Now that you have it installed, in your main file, add this following code to import the project.

```
const NeuralNetwork = require('neural-network-node')
```

### Browser

Browser support is finally here!

To get started, create a script such as that follows in order to import the module from static hosting:

```
<script type="module">
        import { Standard } from "https://unpkg.com/neural-network-node@1.2.9/src/html/index.js";
</script>
```


Once you're done with this, you can access the imported classes at any time in any of the script files. HOWEVER, you have to make sure to import the correct class. Ex: import {Standard}. If you're using DFF, then change this to import {DFF}.



### Java

Coming soon.



# Quick Start Standard



### Creating a neural network


The function to create a new neural network is just:

```
const nn = new NeuralNetwork.Standard(input_nodes, hidden_nodes, output_nodes);
```

In this case, you will have to change some of these.
1. replace the ```input_nodes``` with however many inputs you give the network
2. replace the ```hidden_nodes``` with however many hidden nodes you want. this number is arbitrary. You can choose whatever you want but might have to tweak it for optimal results.
3. replace the ```output_nodes``` with however many outputs you want.


Once you're done with this, you have finished creating the neural network. The next step is to train it with data.



### Training the model

To train the model, use the following function:

```
nn.train(input, output)
```

where ```input``` is the value that you input and ```output``` is the value the computer should output.



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
1. replace the ```input_nodes``` with however many inputs you give the network
2. replace the ```hidden_nodes``` with an array. The first element in the array is the number of hidden nodes for the first hidden layer, the second element is for the second layer, and so on.
3. replace the ```output_nodes``` with however many outputs you want.


Once you're done with this, you have finished creating the neural network. The next step is to train it with data.


### Training the model

To train the model, use the following function:

```
nn.train(input, output)
```

where ```input``` is the value that you input and ```output``` is the value the computer should output.


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



```nn.getWeights()```


Returns all of the current weights of the neural network


```nn.getBias()```

Returns all of the current biases of the neural network


```nn.getLearningRate()```

Returns the current learning rate of the neural network


```nn.setWeights()```

Parameters     | What it is    | Required
-----------    | ------------- | --------
weights_array  | Array of weights | Yes


```nn.setBias()```

Parameters     | What it is    | Required
-----------    | --------------- | ---------
bias_array     | Array of biases | Yes

```nn.setLearningRate()```

Parameters     | What it is    | Required
-----------    | -------------- | -------
learning_rate  | New learning rate  | Yes


```nn.predict()```

Parameters    |  What it is     | Required
----------    | --------------- | ---------
input_array   | Array of input data that matches the number of input_nodes | Yes

```nn.train()```


Parameters    |   What it is    | Required
----------    | --------------- | --------
input_array   |  the input data | Yes
output_array  |  what the computer should output | Yes


# Versions

Current Version: 1.2.9

Stable Versions:

**1.3.0** - Added DFF for browser support

**1.2.9** - Added browser support

**1.2.1** - Succesfully added Deep Feed Forward Network. Removed error message for mismatched input data client-side.

**1.1.7** - Failed to add Perceptron support. Forced to remove it temporarily

**1.1.0** - First stable version
