part of "neural_network.dart";

class Neuron {
  static double _sigmoid(double x) => 1.0 / (1.0 + exp(-x));
  static double _sigmoidDerivative(double x) => _sigmoid(x) * (1.0 - _sigmoid(x));
  static double _sigmoidishDerivative(double x) => x * (1.0 - x);

  static double _tanh(double x) => (exp(x) - exp(-x)) / (exp(x) + exp(-x));
  static double _tanhDerivative(double x) => 1.0 - (x * x);

  static double _relu(double x) => x > 0.0 ? x : 0.0;
  static double _reluDerivative(double x) => x > 0.0 ? 1.0 : 0.0;

  static double _leakyRelu(double x) => x > 0.0 ? x : 0.1 * x;
  static double _leakyReluDerivative(double x) => x > 0.0 ? 1.0 : 0.1;

  List<double> weights;
  List<double> weightAdj;
  double delta = 0.0;
  double error = 0.0;
  List<double> inputs;
  double output = 0.0;
  double learningRate;
  ActivationFunction _activationFunction = ActivationFunction.leakyRelu;
  double Function(double) _normalize = _leakyRelu;
  double Function(double) _normalizeDerivative = _leakyReluDerivative;

  /// Returns the current activation function
  ActivationFunction get activationFunction => _activationFunction;

  /// Update normalization and normalizationDerivative methods
  set activationFunction(ActivationFunction af) {
    _activationFunction = af;
    switch (af) {
      case ActivationFunction.leakyRelu:
        _normalize = _leakyRelu;
        _normalizeDerivative = _leakyReluDerivative;
        break;
      case ActivationFunction.relu:
        _normalize = _relu;
        _normalizeDerivative = _reluDerivative;
        break;
      case ActivationFunction.sigmoid:
        _normalize = _sigmoid;
        _normalizeDerivative = _sigmoidDerivative;
        break;
      case ActivationFunction.sigmoidish:
        _normalize = _sigmoid;
        _normalizeDerivative = _sigmoidishDerivative;
        break;
      case ActivationFunction.tanh:
        _normalize = _tanh;
        _normalizeDerivative = _tanhDerivative;
        break;
    }
  }

  Neuron(
    int inputCount, {
    double learningRate,
    ActivationFunction activationFunction,
  }) {
    this.weights = List<double>();
    this.weightAdj = List<double>();
    this.inputs = List<double>();

    // Initialize weights and empty weight adjustments
    for (int j = 0; j < inputCount; j++) {
      weights.add(2 * Network.r.nextDouble() - 1);
      weightAdj.add(0.0);
    }
    this.learningRate = learningRate ?? _defaultLearningRate;
    this.activationFunction = activationFunction ?? _defaultActivationFunction;
  }

  factory Neuron.fromJson(Map<String, dynamic> map) {
    Neuron n = Neuron(0);
    n.weights = map["weights"];
    n.weightAdj = map["weightsAdj"];
    n.inputs = map["inputs"];
    n.error = map["error"];
    n.output = map["output"];
    n.delta = map["delta"];
    n.learningRate = map["learningRate"];
    n.activationFunction = ActivationFunction.values[map["activationFunction"]];
    return n;
  }

  Map<String, dynamic> toJson() {
    var output = {
      "weights": weights.map<double>((n) => n.isNaN ? 1 : n).toList(),
      "weightsAdj": weightAdj?.map<double>((n) => n.isNaN ? 0 : n)?.toList() ?? [],
      "inputs": inputs?.map<double>((n) => n.isNaN ? 1 : n)?.toList() ?? [],
      "error": (error?.isNaN) ?? true ? 0 : error,
      "output": this.output?.isNaN ?? true ? 0 : this.output,
      "delta": delta?.isNaN ?? true ? 0 : delta,
      "learningRate": this.learningRate,
      "activationFunction": ActivationFunction.values.indexOf(this.activationFunction ?? ActivationFunction.sigmoid),
    };
    return output;
  }

  void reset() {
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (2 * Network.r.nextDouble() - 1);
    }
  }

  void mutate() {
    for (int i = 0; i < this.weights.length; i++) {
      weights[i] += (2 * Network.r.nextDouble() - 1) * Network.mutationFactor;
    }
  }

  

  double _softplus(double x) {
    var val = log(1 + exp(x));
    if (val.isNaN || val.isInfinite) {
      return (2 * Network.r.nextDouble() - 1);
    }
    return val;
  }

  double _softplusDerivative(double x) {
    var val = 1.0 / (1.0 + exp(-x));
    return val;
  }

  ///Adjust weights to comply with a given input
  void _adjustForInput(List<double> input) {
    // Add the bias to the input
    this.inputs = [1.0].followedBy(input).toList();

    // If we already adjusted exit
    if (this.weights.length == this.inputs.length) return;

    // Adjust if input is too large
    if (this.inputs.length >= weights.length) {
      for (int i = weights.length; i < this.inputs.length; i++) {
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
    // Adjust the weights based on the input and add bias
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

  void backPropagationHidden(List<double> nextLayerDeltas, List<double> nextLayerWeights) {
    delta = 0;
    // Pulling each weight corresponding to this neuron in the layer
    for (int j = 0; j < nextLayerDeltas.length; j++) {
      delta += nextLayerDeltas[j] * nextLayerWeights[j];
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
      weights[i] += weightAdj[i] * learningRate;
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
  // softplus,
}
