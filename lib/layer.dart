part of "neural_network.dart";

class Layer {
  static Random r = Random();

  List<Neuron> neurons;

  List<List<double>> get weights => this.neurons.map<List<double>>((n) => n.weights).toList();

  /// Flip the weights so they are weights[weightIndex][neuronIndex]
  List<List<double>> get weightsByNeuron {
    List<List<double>> ws = List<List<double>>();
    List<List<double>> normalWeights = this.weights;
    for (int i = 0; i < this.neurons[0].weights.length; i++) {
      ws.add(List<double>());
    }
    // Flip the weights so they are weights[weightIndex][neuronIndex]
    for (int i = 0; i < this.neurons.length; i++) {
      for (int j = 0; j < this.neurons[0].weights.length; j++) {
        ws[j].add(normalWeights[i][j]);
      }
    }

    return ws;
  }

  List<double> get delta => this.neurons.map<double>((n) => n.delta).toList();
  List<double> get outputs => this.neurons.map<double>((n) => n.output).toList();

  Layer(int inputCount, int outputCount) {
    this.neurons ??= List<Neuron>();

    // Add a new list of weights for each neuron
    for (int i = 0; i < outputCount; i++) {
      this.neurons.add(Neuron(inputCount));
    }
  }

  factory Layer.fromJson(Map<String, dynamic> map) {
    Layer l = Layer(0, 0);
    l.neurons = map["neurons"].map<Neuron>((nString) => Neuron.fromJson(nString)).toList();
    return l;
  }

  Map<String, dynamic> toJson() {
    return {
      "neurons": this.neurons.map<Map<String, dynamic>>((n) => n.toJson()).toList(),
    };
  }

  void resize(int newSize) {
    if (newSize > this.neurons.length) {
      for (int i = this.neurons.length; i < newSize; i++) {
        this.neurons.add(Neuron(this.neurons[0].weights.length));
      }
    } else {
      this.neurons = this.neurons.take(newSize).toList();
    }
  }

  List<double> forwardPropagation(List<double> inputs) {
    neurons.forEach((n) => n.forwardPropagation(inputs));

    return outputs;
  }

  void backPropagationOutput(List<double> expected) {
    for (int i = 0; i < neurons.length; i++) {
      neurons[i].backPropagationOutput(expected[i]);
    }
  }

  void backPropagationHidden(Layer nextLayer) {
    for (int i = 0; i < neurons.length; i++) {
      neurons[i].backPropagationHidden(nextLayer.delta, nextLayer.weightsByNeuron[i]);
    }
  }

  void updateWeights() {
    for (Neuron neuron in neurons){
      neuron.applyAdjustments();
    }
  }
}
