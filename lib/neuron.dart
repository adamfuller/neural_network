part of "neural_network.dart";

class Neuron {
  List<double> weights;
  List<double> weightAdj;
  double delta = 0.0;
  double error = 0.0;
  List<double> inputs;
  double output = 0.0;

  Neuron(int inputCount) {
    this.weights = List<double>();
    this.weightAdj = List<double>();
    this.inputs = List<double>();

    // Initialize weights and empty weight adjustments
    for (int j = 0; j < inputCount; j++) {
      weights.add(2 * Network.r.nextDouble() - 1);
      weightAdj.add(0.0);
    }
  }

  factory Neuron.fromJson(Map<String, dynamic> map) {
    Neuron n = Neuron(0);
    n.weights = map["weights"];
    n.weightAdj = map["weightAdj"];
    n.inputs = map["inputs"];
    n.error = map["error"];
    n.output = map["output"];
    n.delta = map["delta"];
    return n;
  }

  Map<String, dynamic> toJson() {
    var output = {
      "weights": weights.map<double>((n) => n.isNaN ? 1 : n).toList(),
      "weightAdj": weightAdj?.map<double>((n) => n.isNaN ? 0 : n)?.toList() ?? [],
      "inputs": inputs?.map<double>((n) => n.isNaN ? 1 : n)?.toList() ?? [],
      "error": (error?.isNaN) ?? true ? 0 : error,
      "output": this.output?.isNaN ?? true ? 0 : this.output,
      "delta": delta?.isNaN ?? true ? 0 : delta,
    };
    return output;
  }

  void reset() {
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (2 * Network.r.nextDouble() - 1);
    }
  }

  double _sigmoid(double x) => 1.0 / (1.0 + exp(-x));
  double _sigmoidDerivative(double x) => _sigmoid(x) * (1.0 - _sigmoid(x));
  double _sigmoidishDerivative(double x) => x * (1.0 - x);

  double _tanh(double x) => (exp(x) - exp(-x)) / (exp(x) + exp(-x));
  double _tanhDerivative(double x) => 1.0 - (x * x);

  double _relu(double x) => x > 0.0 ? x : 0.0;
  double _reluDerivative(double x) => x > 0.0 ? 1.0 : 0.0;

  double _leakyRelu(double x) => x > 0.0 ? x : 0.1 * x;
  double _leakyReluDerivative(double x) => x > 0.0 ? 1.0 : 0.1;

  // double _softplus(double x) {
  //   var val = log(1 + exp(x));
  //   if (val.isNaN || val.isInfinite) {
  //     return (2 * Network.r.nextDouble() - 1);
  //   }
  //   return val;
  // }

  // double _softplusDerivative(double x) {
  //   var val = 1.0 / (1.0 + exp(-x));
  //   return val;
  // }

  double _normalize(double x) {
    switch (Network.activationFunction) {
      case ActivationFunction.relu:
        return _relu(x);
      case ActivationFunction.leakyRelu:
        return _leakyRelu(x);
      case ActivationFunction.sigmoid:
        return _sigmoid(x);
      case ActivationFunction.sigmoidish:
        return _sigmoid(x);
      case ActivationFunction.tanh:
        return _tanh(x);
      // case ActivationFunction.softplus:
      //   return _softplus(x);
      default:
        return _sigmoid(x);
    }
  }

  double _normalizeDerivative(double x) {
    switch (Network.activationFunction) {
      case ActivationFunction.relu:
        return _reluDerivative(x);
      case ActivationFunction.leakyRelu:
        return _leakyReluDerivative(x);
      case ActivationFunction.sigmoid:
        return _sigmoidDerivative(x);
      case ActivationFunction.tanh:
        return _tanhDerivative(x);
      // case ActivationFunction.softplus:
      //   return _softplusDerivative(x);
      case ActivationFunction.sigmoidish:
        return _sigmoidishDerivative(x);
      default:
        return _sigmoidDerivative(x);
    }
  }

  ///Adjust weights to comply with a given input
  void _adjustForInput(List<double> input) {
    this.inputs = input;
    if (this.weights.length == input.length) return;

    // Adjust if input is too large
    if (input.length >= weights.length) {
      for (int i = weights.length; i < input.length; i++) {
        weights.add((2 * Network.r.nextDouble() - 1));
        weightAdj.add(0.0);
      }
    }
    // Adjust if input is too small
    if (inputs.length < weights.length) {
      weights = weights.take(weights.length - (weights.length - inputs.length)).toList();
      weightAdj = weightAdj.take(weightAdj.length - (weightAdj.length - inputs.length)).toList();
    }
  }

  /// Calculate the output of this neuron for a given input
  double forwardPropagation(List<double> input) {
    output = 0;
    // Adjust the weights based on the input
    _adjustForInput(input);

    //
    // Calculate the output
    //
    for (int i = 0; i < inputs.length; i++) {
      output += inputs[i] * weights[i];
    }

    // Normalize the output
    output = _normalize(output);
    return output;
  }

  /// Adjust output neuron based on an expected output
  void backPropagationOutput(double expectedOutput) {
    // Difference between expected and calculated output
    error = expectedOutput - output;

    // Adjust for input based on error * gradient of normalized(output);
    delta = error * _normalizeDerivative(output);

    if (delta.isNaN) {
      print("NAN delta!");
    }

    // For each input calculate the new corresponding weight
    weightAdj = inputs.map<double>((i) => i * delta).toList();
  }

  void backPropagationHidden(List<double> deltaForward, List<double> weightsForward) {
    delta = 0;
    // Pulling each weight corresponding to this neuron in the layer
    for (int j = 0; j < deltaForward.length; j++) {
      delta += deltaForward[j] * weightsForward[j];
    }
    delta *= _normalizeDerivative(output);

    // Assign the adjustments
    for (int i = 0; i < inputs.length; i++) {
      if (weightAdj.length <= i) weightAdj.add(0.0);
      weightAdj[i] = delta * inputs[i];
    }
  }

  void applyAdjustments() {
    double maxWeight = 0;
    for (int i = 0; i < this.weights.length; i++) {
      weights[i] += weightAdj[i] * Network.learningFactor;
      if (weights[i].abs() > maxWeight) maxWeight = weights[i].abs();
    }
    // normalize the weights??
    for (int i = 0; i < weights.length; i++) {
      weights[i] /= maxWeight;
    }
  }
}

enum ActivationFunction {
  leakyRelu,
  relu,
  sigmoid,
  sigmoidish,
  tanh,
}
