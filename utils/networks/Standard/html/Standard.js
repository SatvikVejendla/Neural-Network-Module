import Matrix from "../../../global/html/Matrix.js";
import Sigmoid from "../../../global/html/Sigmoid.js";

export default class Standard {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();
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
    let inputs = Matrix.fromArray(input_array);

    //input to hidden
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);

    //activation
    hidden.map(Sigmoid.sigmoid);

    //hidden to output
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(Sigmoid.sigmoid);

    //converting matrix to array and return
    return output.toArray();
  }

  train(input_array, target_array) {
    let inputs = Matrix.fromArray(input_array);

    //input to hidden
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);

    //activation
    hidden.map(Sigmoid.sigmoid);

    //hidden to output
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(Sigmoid.sigmoid);

    let targets = Matrix.fromArray(target_array);
    //error calculation
    let output_errors = Matrix.subtract(targets, outputs);
    let gradients = Matrix.map(outputs, Sigmoid.dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    //Weights and Biases adjustments
    this.weights_ho.add(weight_ho_deltas);
    this.bias_o.add(gradients);

    //hidden layer errors
    let who_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, output_errors);
    let hidden_gradient = Matrix.map(hidden, Sigmoid.dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hidden_gradient);
  }
}
