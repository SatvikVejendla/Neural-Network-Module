const NeuralNetwork = require("neural-network-node");

let nn = new NeuralNetwork.Standard(2, 4, 1);

/*
   XOR is when you want one value to be True and another to be False. 
   So, for example, True and False would return True in XOR. 
   However, True and True would return False.*/
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

function random(x) {
  return x[Math.random() * x.length];
}

for (let i = 0; i < 1000; i++) {
  let data = random(training_data);
  nn.train(data.inputs, data.outputs);
}

const output = nn.predict([0, 1]); //The more you train the model, the closer this gets to one
console.log(output);
