

class Standard {
  private int input_nodes;
  private int hidden_nodes;
  private int output_nodes;
  private Matrix weights_ih;
  private Matrix weights_ho;
  private double learning_rate;
  private Matrix bias_h;
  private Matrix bias_o;

  public Standard(int input_nodes, int hidden_nodes, int output_nodes) {
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

  public double sigmoid(int x){
      return 1/(1+Math.exp(-x));
  }
  public double dsigmoid(int y){
      return y*(1-y);
  }

  public double getLearningRate() {
    return this.learning_rate;
  }

  public void setLearningRate(int x) {
    this.learning_rate = x;
  }

  public void setWeights(Matrix[] x) {
    this.weights_ih = x[0];
    this.weights_ho = x[1];
  }

  public void setBiases(Matrix[] x) {
    this.biases_h = x[0];
    this.biases_o = x[1];
  }

  public Matrix getWeights(int x) {
    if (x == 0) {
      return this.weights_ih;
    } else {
      return this.weights_ho;
    }
  }
  public Matrix getBias(int x) {
    if (x == 0) {
      return this.bias_h;
    } else {
      return this.bias_o;
    }
  }

  public int predict(double[] input_array) {
    Matrix inputs = Matrix.fromArray(input_array);

    //input to hidden
    Matrix hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);

    //activation
    hidden.map(sigmoid);

    //hidden to output
    Matrix output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(sigmoid);

    //converting matrix to array and return
    return output.toArray();
  }

  public void train(input_array, target_array) {
    Matrix inputs = Matrix.fromArray(input_array);

    //input to hidden
    Matrix hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);

    //activation
    hidden.map(sigmoid);

    //hidden to output
    Matrix outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(sigmoid);

    Matrix targets = Matrix.fromArray(target_array);
    //error calculation
    Matrix output_errors = Matrix.subtract(targets, outputs);
    Matrix gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    Matrix hidden_T = Matrix.transpose(hidden);
    Matrix weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    //Weights and Biases adjustments
    this.weights_ho.add(weight_ho_deltas);
    this.bias_o.add(gradients);

    //hidden layer errors
    Matrix who_t = Matrix.transpose(this.weights_ho);
    Matrix hidden_errors = Matrix.multiply(who_t, output_errors);
    Matrix hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    Matrix inputs_T = Matrix.transpose(inputs);
    Matrix weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hidden_gradient);
  }
}

