let defaultConfig = {
  learning_rate: 0.1,
  activation: "sigmoid",
};

class DFF {
  constructor(nodes, config = defaultConfig) {
    this.input_nodes = nodes[0];
    this.hidden_nodes = nodes.splice(1, nodes.length - 1);
    let hidden_nodes = this.hidden_nodes;
    this.output_nodes = nodes[nodes.length - 1];

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
        this.weights["weights_h" + i] = new Matrix(
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

    this.learning_rate = config.learning_rate;

    if (config.activation.toLowerCase() == "sigmoid") {
      this.activation = {
        function: np.sigmoid,
        dfunction: np.dsigmoid,
      };
    } else if (config.activation.toLowerCase() == "tanh") {
      this.activation = {
        function: np.tanh,
        dfunction: np.dtanh,
      };
    } else if (config.activation.toLowerCase() == "relu") {
      this.activation = {
        function: np.relu,
        dfunction: np.heaviside,
      };
    } else if (config.activation.toLowerCase() == "leakyrelu") {
      this.activation = {
        function: np.leakyrelu,
        dfunction: np.dleakyrelu,
      };
    }
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

  setState(json) {
    this.input_nodes = json.layers.input_nodes;
    this.hidden_nodes = json.layers.hidden_nodes;
    this.output_nodes = json.layers.output_nodes;
    this.weights = {
      weights_ih: new Matrix(this.hidden_nodes[0], this.input_nodes),
    };
    for (let i = 0; i < this.hidden_nodes.length; i++) {
      if (i == this.hidden_nodes.length - 1) {
        this.weights["weights_ho"] = new Matrix(
          this.output_nodes,
          this.hidden_nodes[i]
        );
        this.weights["weights_ho"].data = json.weights["weights_ho"].data;
      } else if (i >= 0) {
        this.weights["weights_h" + i] = new Matrix(
          this.hidden_nodes[i + 1],
          this.hidden_nodes[i]
        );
        this.weights["weights_h" + i].data = json.weights["weights_h" + i].data;
      }
    }
    this.weights = json.weights;
    this.biases = json.biases;
    this.learning_rate = json.learning_rate;
  }

  predict(input_array) {
    let inputs = Matrix.fromArray(input_array);
    //input to hidden
    let hidden = Matrix.multiply(this.weights["weights_ih"], inputs);
    hidden.add(this.biases["bias_ih"]);

    //activation
    hidden.map(this.activation.function);

    let tempoutput = hidden;
    for (let i = 0; i < Object.keys(this.weights).length - 2; i++) {
      tempoutput = Matrix.multiply(this.weights["weights_h" + i], tempoutput);
      tempoutput.add(this.biases["bias_h" + i]);
      tempoutput.map(this.activation.function);
    }

    //hidden to output
    let output = Matrix.multiply(this.weights["weights_ho"], tempoutput);
    output.add(this.biases["bias_ho"]);
    output.map(this.activation.function);

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
    hidden.map(this.activation.function);
    values["ih"] = hidden;

    let tempoutput = hidden;
    for (let i = 0; i < Object.keys(this.weights).length - 2; i++) {
      tempoutput = Matrix.multiply(this.weights["weights_h" + i], tempoutput);
      tempoutput.add(this.biases["bias_h" + i]);
      tempoutput.map(this.activation.function);
      values["h" + i] = tempoutput;
    }

    let output = Matrix.multiply(this.weights["weights_ho"], tempoutput);
    output.add(this.biases["bias_ho"]);
    output.map(this.activation.function);
    values["ho"] = output;

    //error calculation
    let output_errors = Matrix.subtract(targets, output);
    let gradients = Matrix.map(output, this.activation.dfunction);
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
        this.activation.dfunction
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
    let hidden_gradient = Matrix.map(hidden, this.activation.dfunction);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);
    this.weights["weights_ih"].add(weight_ih_deltas);
    this.biases["bias_ih"].add(hidden_gradient);
  }
}
