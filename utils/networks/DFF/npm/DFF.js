const Matrix = require("../../../global/npm/Matrix");
const Sigmoid = require("../../../global/npm/Sigmoid");

class DFF {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights = {
      weights_ih: new Matrix(this.hidden_nodes[0], this.input_nodes),
    };
    for (let i = 0; i < hidden_nodes.length; i++) {
      if (i == hidden_nodes.length - 1) {
        this.weights["weights_ho"] = new Matrix(
          this.output_nodes,
          this.hidden_nodes[i]
        );
      } else if (i >= 0) {
        this.weights["weights_h" + i] = new Matrix(
          this.hidden_nodes[i + 1],
          this.hidden_nodes[i]
        );
      }
    }
    for (let i in this.weights) {
      this.weights[i].randomize();
    }

    this.biases = {
      bias_ih: new Matrix(this.hidden_nodes[0], 1),
    };

    for (let i = 0; i < hidden_nodes.length; i++) {
      if (i == hidden_nodes.length - 1) {
        this.biases["bias_ho"] = new Matrix(this.output_nodes, 1);
      } else {
        this.biases["bias_h" + i] = new Matrix(this.hidden_nodes[i + 1], 1);
      }
    }
    for (let i in this.biases) {
      this.biases[i].randomize();
    }

    this.learning_rate = 0.1;
  }
  getWeights() {
    return this.weights;
  }
  getBias() {
    return this.biases;
  }

  predict(input_array) {
    let inputs = Matrix.fromArray(input_array);
    //input to hidden
    let hidden = Matrix.multiply(this.weights["weights_ih"], inputs);
    hidden.add(this.biases["bias_ih"]);

    //activation
    hidden.map(Sigmoid.sigmoid);

    let tempoutput = hidden;
    for (let i = 0; i < Object.keys(this.weights).length - 2; i++) {
      tempoutput = Matrix.multiply(this.weights["weights_h" + i], tempoutput);
      tempoutput.add(this.biases["bias_h" + i]);
      tempoutput.map(Sigmoid.sigmoid);
    }

    //hidden to output
    let output = Matrix.multiply(this.weights["weights_ho"], tempoutput);
    output.add(this.biases["bias_ho"]);
    output.map(Sigmoid.sigmoid);

    //converting matrix to array and return
    return output.toArray();
  }

  train(input_array, target_array) {
    let inputs = Matrix.fromArray(input_array);
    let targets = Matrix.fromArray(target_array);

    let values = {};
    //guess
    let hidden = Matrix.multiply(this.weights["weights_ih"], inputs);
    hidden.add(this.biases["bias_ih"]);
    hidden.map(Sigmoid.sigmoid);
    values["ih"] = hidden;

    let tempoutput = hidden;
    for (let i = 0; i < Object.keys(this.weights).length - 2; i++) {
      tempoutput = Matrix.multiply(this.weights["weights_h" + i], tempoutput);
      tempoutput.add(this.biases["bias_h" + i]);
      tempoutput.map(Sigmoid.sigmoid);
      values["h" + i] = tempoutput;
    }

    let output = Matrix.multiply(this.weights["weights_ho"], tempoutput);
    output.add(this.biases["bias_ho"]);
    output.map(Sigmoid.sigmoid);
    values["ho"] = output;

    //error calculation
    let output_errors = Matrix.subtract(targets, output);
    let gradients = Matrix.map(output, Sigmoid.dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);
    this.weights["weights_ho"].add(weight_ho_deltas);
    this.biases["bias_ho"].add(gradients);

    let last_error = output_errors;
    for (let i = Object.keys(this.weights).length - 2; i > 0; i--) {
      let temp_who_t;
      if (i == Object.keys(this.weights).length - 2) {
        temp_who_t = Matrix.transpose(this.weights["weights_ho"]);
      } else {
        temp_who_t = Matrix.transpose(this.weights["weights_h" + i]);
      }
      let temp_hidden_errors = Matrix.multiply(temp_who_t, last_error);
      last_error = temp_hidden_errors;

      let temp_hidden_gradient = Matrix.map(
        values["h" + (i - 1)],
        Sigmoid.dsigmoid
      );
      temp_hidden_gradient.multiply(temp_hidden_errors);
      temp_hidden_gradient.multiply(this.learning_rate);
      let temp_inputs_T;
      if (i == 1) {
        temp_inputs_T = Matrix.transpose(values["ih"]);
      } else {
        temp_inputs_T = Matrix.transpose(values["h" + (i - 2)]);
      }

      let temp_weight_ih_deltas = Matrix.multiply(
        temp_hidden_gradient,
        temp_inputs_T
      );
      this.weights["weights_h" + (i - 1)].add(temp_weight_ih_deltas);
      this.biases["bias_h" + (i - 1)].add(temp_hidden_gradient);
    }

    //

    let who_t = Matrix.transpose(this.weights["weights_h0"]);
    let hidden_errors = Matrix.multiply(who_t, last_error);
    let hidden_gradient = Matrix.map(hidden, Sigmoid.dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);
    this.weights["weights_ih"].add(weight_ih_deltas);
    this.biases["bias_ih"].add(hidden_gradient);
  }
}
