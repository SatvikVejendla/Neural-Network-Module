# Neural-Network-Node
A simple easy to use Node JS module which lets you create your own neural network and train it.




# Documentation


### Set up

Begin by installing the node module with this command:

```npm install neural-network-node```

Now that you have it installed, in your main file, add this following code to import the project.

```
const NeuralNetwork = require('neural-network-node')
```



### Creating a neural network


The function to create a new neural network is just:

```
const nn = new NeuralNetwork(input_nodes, hidden_nodes, output_nodes);
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
nn.feedforward(input)
```

This will make the computer use it's previous tested data to make a guess for what the output should be.
