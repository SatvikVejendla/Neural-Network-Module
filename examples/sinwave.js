const NeuralNetwork = require("neural-network-node");

let nn = new NeuralNetwork(1, 2, 1);

function generatedata() {
  let input = Math.random() * 2 * Math.PI;
  let output = Math.sin(input);

  return {
    input: [input],
    output: [output],
  };
}

for (let i = 0; i < 10000; i++) {
  var data = generatedata();
  nn.train(data.input, data.output);
}

console.log("Finished training");

const output = nn.predict(Math.PI); //The more you train the model, the closer this gets to zero
console.log(output);
