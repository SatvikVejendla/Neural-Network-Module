const NN = require("./src/npm/index.js");

const config = {
  learning_rate: 0.1,
  activation: "relu",
};
let nn = new NN.Standard([2, 3, 1], config);

for (let i = 0; i < 2; i++) {
  nn.train([0, 0], [1]);
  nn.train([0, 1], [0]);
  nn.train([1, 0], [0]);
}
console.log(nn.predict([0, 0]));
