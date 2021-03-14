const np = require("numpy-matrix-js");
const Sigmoid = require("../../../global/npm/Sigmoid");

class Standard {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights_ih = np.random.rand(this.hidden_nodes, this.input_nodes);
    this.weights_ho = np.random.rand(this.output_nodes, this.hidden_nodes);

    this.bias_h = np.random.rand(this.hidden_nodes, 1);
    this.bias_o = np.random.rand(this.output_nodes, 1);

    this.learning_rate = 0.1;
  }

  getLearningRate() {
    return this.learning_rate;
  }

  setLearningRate(x = 0.1) {
    this.learning_rate = x;
  }

  setWeights(x) {
    this.weights_ih = x[0];
    this.weights_ho = x[1];
  }

  setBiases(x) {
    this.biases_h = x[0];
    this.biases_o = x[1];
  }

  getWeights(x) {
    if (x == 0) {
      return this.weights_ih;
    } else {
      return this.weights_ho;
    }
  }
  getBias(x) {
    if (x == 0) {
      return this.bias_h;
    } else {
      return this.bias_o;
    }
  }

  predict(input_array) {
    let inputs = np.zeros(input_array.length, 1);
    for (let i = 0; i < input_array.length; i++) {
      inputs[i][0] = input_array[i];
    }
    //input to hidden
    let hidden = np.matmul(this.weights_ih, inputs);
    hidden = np.add(hidden, this.bias_h);

    //activation
    hidden.map(Sigmoid.sigmoid);

    //hidden to output
    let output = np.matmul(this.weights_ho, hidden);
    output = np.add(output, this.bias_o);
    output.map(Sigmoid.sigmoid);

    //converting matrix to array and return
    return output;
  }

  train(input_array, target_array) {
    let inputs = np.zeros(input_array.length, 1);
    for (let i = 0; i < input_array.length; i++) {
      inputs[i][0] = input_array[i];
    }
    //input to hidden
    let hidden = np.matmul(this.weights_ih, inputs);
    hidden = np.add(hidden, this.bias_h);

    //activation
    hidden.map(Sigmoid.sigmoid);

    //hidden to output
    let outputs = np.matmul(this.weights_ho, hidden);
    outputs = np.add(outputs, this.bias_o);
    outputs.map(Sigmoid.sigmoid);

    let targets = np.zeros(target_array.length, 1);
    for (let i = 0; i < target_array.length; i++) {
      targets[i][0] = target_array[i];
    }
    //error calculation
    let output_errors = np.subtract(targets, outputs);
    let gradients = outputs;
    gradients.map(Sigmoid.dsigmoid);
    gradients = np.matmul(gradients, output_errors);
    gradients.multiply(this.learning_rate);

    let hidden_T = np.transpose(hidden);
    let weight_ho_deltas = np.matmul(gradients, hidden_T);

    //Weights and Biases adjustments
    this.weights_ho = np.add(this.weights_ho, weight_ho_deltas);
    this.bias_o = np.add(this.bias_o, gradients);

    //hidden layer errors
    let who_t = np.transpose(this.weights_ho);
    let hidden_errors = np.matmul(who_t, output_errors);
    let hidden_gradient = hidden;
    hidden_gradient.map(Sigmoid.dsigmoid);
    hidden_gradient = np.matmul(hidden_gradient, hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    let inputs_T = np.transpose(inputs);
    let weight_ih_deltas = np.matmul(hidden_gradient, inputs_T);

    this.weights_ih = np.add(this.weights_ih, weight_ih_deltas);
    this.bias_h = np.add(this.bias_h, hidden_gradient);
  }
}

module.exports = Standard;
