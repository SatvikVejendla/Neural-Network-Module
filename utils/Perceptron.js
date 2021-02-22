class Perceptron {
  constructor(n, learningrate) {
    this.weights = new Array(n);
    this.weights[0] = -0.12196458643444202;
    this.weights[1] = 0.4085250337315881;
    this.weights[2] = -0.16474934143118125;
    this.learningrate = learningrate;
  }

  train(inputs, target) {
    let guess = this.guess(inputs);
    let error = target - guess;

    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] += this.learningrate * error * inputs[i];
    }
  }

  guess(inputs) {
    // Sum all values
    let sum = 0;
    for (let i = 0; i < this.weights.length; i++) {
      sum += inputs[i] * this.weights[i];
    }
    // Result is sign of the sum, -1 or 1
    return this.activate(sum);
  }

  activate(sum) {
    return sum > 0 ? 1 : -1;
  }

  // Return weights
  getWeights() {
    return this.weights;
  }
}

module.exports = Perceptron;
