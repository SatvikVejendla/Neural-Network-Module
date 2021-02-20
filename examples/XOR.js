const NeuralNetwork = require("neural-network-node");

let nn = new NeuralNetwork(2, 4, 1);

let training_data = [
  {
    inputs: [0, 0],
    outputs: [0],
  },
  {
    inputs: [0, 1],
    outputs: [1],
  },
  {
    inputs: [1, 0],
    outputs: [1],
  },
  {
    inputs: [1, 1],
    outputs: [0],
  },
];

for (let i = 0; i < 1000; i++) {
  let data = random(training_data);
  nn.train(data.inputs, data.outputs);
}

const output = nn.predict([0, 1]); //The more you train the model, the closer this gets to one
console.log(output);
