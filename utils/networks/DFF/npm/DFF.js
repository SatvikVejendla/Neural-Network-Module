const np = require("numpy-matrix-js");
const Sigmoid = require("../../../global/npm/Sigmoid");

class DFF {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights = {
      weights_ih: np.random.rand(this.hidden_nodes[0], this.input_nodes),
    };
    for (let i = 0; i < hidden_nodes.length; i++) {
      if (i == hidden_nodes.length - 1) {
        this.weights["weights_ho"] = np.random.rand(
          this.output_nodes,
          this.hidden_nodes[i]
        );
      } else if (i >= 0) {
        this.weights["weights_h" + i] = np.random.rand(
          this.hidden_nodes[i + 1],
          this.hidden_nodes[i]
        );
      }
    }

    this.biases = {
      bias_ih: np.random.rand(this.hidden_nodes[0], 1),
    };

    for (let i = 0; i < hidden_nodes.length; i++) {
      if (i == hidden_nodes.length - 1) {
        this.biases["bias_ho"] = np.random.rand(this.output_nodes, 1);
      } else {
        this.biases["bias_h" + i] = np.random.rand(this.hidden_nodes[i + 1], 1);
      }
    }

    this.learning_rate = 0.1;
  }

  getState() {
    return {
      layers: {
        input_nodes: this.input_nodes,
        hidden_nodes: this.hidden_nodes,
        output_nodes: this.output_nodes,
      },
      weights: this.weights,
      biases: this.biases,
      learning_rate: this.learning_rate,
    };
  }

  predict(input_array) {
    let inputs = np.zeros(input_array.length, 1);
    for (let i = 0; i < input_array.length; i++) {
      inputs[i][0] = input_array[i];
    }
    //input to hidden
    let hidden = np.matmul(this.weights["weights_ih"], inputs);

    hidden = np.add(hidden, this.biases["bias_ih"]);
    //activation
    hidden.map(Sigmoid.sigmoid);

    let tempoutput = hidden;
    for (let i = 0; i < Object.keys(this.weights).length - 2; i++) {
      tempoutput = np.matmul(this.weights["weights_h" + i], tempoutput);
      tempoutput = np.add(tempoutput, this.biases["bias_h" + i]);
      tempoutput.map(Sigmoid.sigmoid);
    }

    //hidden to output
    let output = np.matmul(this.weights["weights_ho"], tempoutput);
    output = np.add(output, this.biases["bias_ho"]);
    output.map(Sigmoid.sigmoid);

    //converting matrix to array and return
    return output;
  }

  train(input_array, target_array) {
    let inputs = np.zeros(input_array.length, 1);
    for (let i = 0; i < input_array.length; i++) {
      inputs[i][0] = input_array[i];
    }
    let targets = np.zeros(target_array.length, 1);
    for (let i = 0; i < target_array.length; i++) {
      targets[i][0] = target_array[i];
    }

    let values = {};
    //guess
    let hidden = np.matmul(this.weights["weights_ih"], inputs);
    hidden = np.add(hidden, this.biases["bias_ih"]);
    hidden.map(Sigmoid.sigmoid);
    values["ih"] = hidden;

    let tempoutput = hidden;
    for (let i = 0; i < Object.keys(this.weights).length - 2; i++) {
      tempoutput = np.matmul(this.weights["weights_h" + i], tempoutput);
      tempoutput = np.add(tempoutput, this.biases["bias_h" + i]);
      tempoutput.map(Sigmoid.sigmoid);
      values["h" + i] = tempoutput;
    }

    let output = np.matmul(this.weights["weights_ho"], tempoutput);
    output = np.add(output, this.biases["bias_ho"]);
    output.map(Sigmoid.sigmoid);
    values["ho"] = output;

    //error calculation
    let output_errors = np.subtract(targets, output);
    let gradients = output;
    gradients.map(Sigmoid.dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);
    let hidden_T = np.transpose(hidden);
    let weight_ho_deltas = np.matmul(gradients, hidden_T);
    this.weights["weights_ho"] = np.add(
      this.weights["weights_ho"],
      weight_ho_deltas
    );
    this.biases["bias_ho"] = np.add(this.biases["bias_ho"], gradients);

    let last_error = output_errors;
    for (let i = Object.keys(this.weights).length - 2; i > 0; i--) {
      let temp_who_t;
      if (i == Object.keys(this.weights).length - 2) {
        temp_who_t = np.transpose(this.weights["weights_ho"]);
      } else {
        temp_who_t = np.transpose(this.weights["weights_h" + i]);
      }
      let temp_hidden_errors = np.matmul(temp_who_t, last_error);
      last_error = temp_hidden_errors;
      let temp_hidden_gradient = values["h" + (i - 1)];
      temp_hidden_gradient.map(Sigmoid.dsigmoid);
      temp_hidden_gradient.multiply(temp_hidden_errors);
      temp_hidden_gradient.multiply(this.learning_rate);
      let temp_inputs_T;
      if (i == 1) {
        temp_inputs_T = np.transpose(values["ih"]);
      } else {
        temp_inputs_T = np.transpose(values["h" + (i - 2)]);
      }

      let temp_weight_ih_deltas = np.matmul(
        temp_hidden_gradient,
        temp_inputs_T
      );
      this.weights["weights_h" + (i - 1)] = np.add(
        this.weights["weights_h" + (i - 1)],
        temp_weight_ih_deltas
      );
      this.biases["bias_h" + (i - 1)] = np.add(
        this.biases["bias_h" + (i - 1)],
        temp_hidden_gradient
      );
    }

    //

    let who_t = np.transpose(this.weights["weights_h0"]);
    let hidden_errors = np.matmul(who_t, last_error);
    let hidden_gradient = hidden;
    hidden_gradient.map(Sigmoid.dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);
    let inputs_T = np.transpose(inputs);
    let weight_ih_deltas = np.matmul(hidden_gradient, inputs_T);
    this.weights["weights_ih"] = np.add(
      this.weights["weights_ih"],
      weight_ih_deltas
    );
    this.biases["bias_ih"] = np.add(this.biases["bias_ih"], hidden_gradient);
  }
}

module.exports = DFF;
