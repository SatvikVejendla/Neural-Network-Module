const NeuralNetwork = require("neural-network-node");

let nn = new NeuralNetwork.DFF(1, [3, 2], 1);

/*approximates the y = sinx function.
After you train it, you should be able to input an x-value and get a y-value close to the output*/
function generatedata() {
  let input = Math.random() * 2 * Math.PI;
  let output = Math.sin(input);

  return {
    input: [input],
    output: [output],
  };
}

for (let i = 0; i < 100000; i++) {
  var data = generatedata();
  nn.train(data.input, data.output);
}

console.log("Finished training");

const output = nn.predict([Math.PI]); //The more you train the model, the closer this gets to zero
console.log(output);
