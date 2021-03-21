const np = require("numpy-matrix-js");

let defaultConfig = {
  learning_rate: 0.1,
  activation: "sigmoid",
};

class Standard {
  constructor(nodes, config = defaultConfig) {
    if (nodes.length == 3) {
      this.input_nodes = nodes[0];
      this.hidden_nodes = nodes[1];
      this.output_nodes = nodes[2];
    } else {
      this.input_nodes = nodes[0];
      this.hidden_nodes = 64;
      this.output_nodes = node[1];
    }

    this.weights_ih = np.random.rand(this.hidden_nodes, this.input_nodes);
    this.weights_ho = np.random.rand(this.output_nodes, this.hidden_nodes);

    this.bias_h = np.random.rand(this.hidden_nodes, 1);
    this.bias_o = np.random.rand(this.output_nodes, 1);

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
      weights: {
        weights_ih: this.weights_ih,
        weights_ho: this.weights_ho,
      },
      biases: {
        bias_ih: this.bias_ih,
        bias_ho: this.bias_ho,
      },
      learning_rate: this.learning_rate,
    };
  }

  setState(json) {
    this.weights_ih = json.weights.weights_ih;
    this.weights_ho = json.weights.weights_ho;
    this.bias_ih = json.biases.bias_ih;
    this.bias_ho = json.biases.bias_ho;
    this.learning_rate = json.learning_rate;
    this.input_nodes = json.layers.input_nodes;
    this.hidden_nodes = json.layers.hidden_nodes;
    this.output_nodes = json.layers.output_nodes;
  }

  getLearningRate() {
    return this.learning_rate;
  }

  setLearningRate(x = 0.1) {
    this.learning_rate = x;
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
    hidden.map(this.activation.function);

    //hidden to output
    let output = np.matmul(this.weights_ho, hidden);
    output = np.add(output, this.bias_o);
    output.map(this.activation.function);

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
    hidden.map(this.activation.function);

    //hidden to output
    let outputs = np.matmul(this.weights_ho, hidden);
    outputs = np.add(outputs, this.bias_o);
    outputs.map(this.activation.function);

    let targets = np.zeros(target_array.length, 1);
    for (let i = 0; i < target_array.length; i++) {
      targets[i][0] = target_array[i];
    }
    //error calculation
    let output_errors = np.subtract(targets, outputs);
    let gradients = outputs;
    gradients.map(this.activation.dfunction);
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
    hidden_gradient.map(this.activation.dfunction);
    hidden_gradient = np.matmul(hidden_gradient, hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    let inputs_T = np.transpose(inputs);
    let weight_ih_deltas = np.matmul(hidden_gradient, inputs_T);

    this.weights_ih = np.add(this.weights_ih, weight_ih_deltas);
    this.bias_h = np.add(this.bias_h, hidden_gradient);
  }
}

module.exports = Standard;
