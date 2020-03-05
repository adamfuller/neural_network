library neural_network;

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';

part 'layer.dart';
part 'neuron.dart';

const double _defaultLearningRate = 0.033;
const ActivationFunction _defaultActivationFunction = ActivationFunction.sigmoid;

class Network {
  //
  // Private static
  //

  //
  // Public static
  //
  static double mutationFactor = 0.0033;
  static Random r = Random();

  //
  // Private fields
  //

  //
  // Public fields
  //
  List<Layer> layers;
  List<int> hiddenLayerNeuronCount;

  /// How many times has this been run under the same state
  int runCount = 0;

  /// Average Percent Error of the last few runs
  double averagePercentError = 0;
  List<double> pastErrors = List<double>();

  /// Outputs from running test data through single forwardProp
  List<List<double>> testOutputs = List<List<double>>();

  /// True if this network is currently being trained
  bool isTraining = false;

  /// True if the network was altered externally during training
  bool wasAltered = false;

  String get jsonString => jsonEncode(this.toJson());
  String get prettyJsonString => JsonEncoder.withIndent('  ').convert(this.toJson());
  String get matrixString => matrix.toString();

  /// Returns the learning rate of the first neuron
  ///
  /// Will be removed once learningRate is unique.
  double get learningRate => this.layers[0].neurons[0].learningRate;

  /// Sets the learning rate of each neuron to the same value
  set learningRate(double val) {
    wasAltered = true;
    this.layers.forEach((layer) => layer.neurons.forEach((n) => n.learningRate = val));
  }

  /// Sets the Activation Function of each neuron to the same one
  set activationFunction(ActivationFunction av) {
    wasAltered = true;
    this.layers.forEach((layer) => layer.neurons.forEach((n) => n.activationFunction = av));
  }

  /// Returns the Activation Function of the first neuron
  ///
  /// Will be removed once activationFunction is unique.
  ActivationFunction get activationFunction => this.layers[0].neurons[0].activationFunction;

  List<List<List<double>>> get matrix {
    List<List<List<double>>> values = List<List<List<double>>>(this.layers.length);
    for (int layerIndex = 0; layerIndex < this.layers.length; layerIndex++) {
      values[layerIndex] = List<List<double>>();
      for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.length; neuronIndex++) {
        values[layerIndex].add(layers[layerIndex].neurons[neuronIndex].weights);
      }
    }
    return values;
  }

  int get maxLayerSize {
    int max = 0;
    for (Layer l in layers) {
      if (l.neurons.length > max) {
        max = l.neurons.length;
      }
    }

    return max;
  }

  Network(
    this.hiddenLayerNeuronCount, {
    ActivationFunction activationFunction = _defaultActivationFunction,
    this.layers,
    double learningRate = _defaultLearningRate,
    this.averagePercentError = 0,
    this.runCount = 0,
    this.testOutputs,
  }) {
    if (this.layers == null) {
      this.layers ??= List<Layer>();
      // Add a layer for each count
      for (int i = 0; i < hiddenLayerNeuronCount.length - 1; i++) {
        layers.add(
          Layer(
            hiddenLayerNeuronCount[i],
            hiddenLayerNeuronCount[i + 1],
          ),
        );
      }
    }
    // Set values for the learning rate of each neuron
    this.learningRate = learningRate ?? _defaultLearningRate;
    // Set each neurons activation function
    this.activationFunction = activationFunction ?? _defaultActivationFunction;
    this.testOutputs = [];
  }

  // static Future<void> _sleep(int milliseconds) async => await Future.delayed(Duration(milliseconds: milliseconds), () => "");

  static Future<Network> train(
    Network network,
    List<List<double>> inputData,
    List<List<double>> outputData, {
    int trainCount = 1,
  }) async {
    if (network.isTraining) return null;
    network.isTraining = true;
    network.wasAltered = false;

    Map<String, dynamic> map = {
      "network": network,
      "inputs": inputData,
      "outputs": outputData,
      "trainCount": trainCount ?? 1,
    };

    Network n = await compute(_trainFromMap, map);

    // If the old network was altered disregard the new one
    if (network.wasAltered) n = network;

    n.isTraining = false;
    n.wasAltered = false;
    return n;
  }

  /// Returns a trained Network
  ///
  /// ```
  ///   Map<String, dynamic> map = {
  ///     "network": Network,           // Network to train
  ///     "inputs": List<List<double>>, // List of training inputs
  ///     "outputs": List<List<double>>,// List of expected outputs
  ///     "trainCount": int,              // # of times to train
  ///   }
  /// ```
  static Network _trainFromMap(Map<String, dynamic> map) {
    if (!map.containsKey("network")) return null;
    if (!map.containsKey("inputs")) return null;
    if (!map.containsKey("outputs")) return null;
    if (!map.containsKey("trainCount")) return null;

    int trainCount = map["trainCount"];
    Network network = map["network"];
    List<List<double>> inputs = map["inputs"];
    List<List<double>> outputs = map["outputs"];

    // Train the network runIndex times
    for (int runIndex = 0; runIndex < trainCount; runIndex++) {
      for (int i = 0; i < inputs.length; i++) {
        network.forwardPropagation(inputs[i]);
        network.backPropagation(outputs[i]);
      }
      network.runCount++;
    }

    // Return the network
    return network;
  }

  factory Network.fromJsonString(String jsonString) {
    Map<String, dynamic> map = jsonDecode(jsonString);
    return Network.fromJson(map);
  }

  factory Network.fromJson(Map<String, dynamic> map) {
    Network n = Network(
      [0],
      layers: map["layers"].map<Layer>((lString) => Layer.fromJson(lString)).toList(),
      averagePercentError: map["averagePercentError"],
      runCount: map["runCount"],
      testOutputs: map["testOutputs"],
      learningRate: map["learningRate"],
      activationFunction: ActivationFunction.values[map["activationFunction"]],
    );
    n.hiddenLayerNeuronCount = map["hiddenLayerNeuronCount"];
    n.pastErrors = map["pastErrors"];

    return n;
  }

  Network copy() => Network.fromJson(this.toJson());

  Map<String, dynamic> toJson() {
    var output = {
      "mutationFactor": Network.mutationFactor,
      "runCount": this.runCount,
      "learningRate": this.learningRate,
      "hiddenLayerNeuronCount": this.hiddenLayerNeuronCount,
      "averagePercentError": this.averagePercentError,
      "pastErrors": this.pastErrors,
      "layers": this.layers?.map<Map<String, dynamic>>((l) => l.toJson())?.toList(),
      "testOutputs": this.testOutputs,
      "activationFunction": ActivationFunction.values.indexOf(this.activationFunction),
    };
    return output;
  }

  void reset() {
    runCount = 0;
    pastErrors.clear();
    averagePercentError = 0;
    wasAltered = true;

    for (Layer layer in layers) {
      for (Neuron neuron in layer.neurons) {
        neuron.reset();
      }
    }
  }

  Network produceMutation() {
    Network copy = Network.fromJson(this.toJson());
    copy.mutate();
    return copy;
  }

  void mutate() {
    this.layers.forEach((l) => l.mutate());
  }

  /// Returns the output of this network for input __inputs__
  List<double> forwardPropagation(List<double> inputs) {
    List<double> output;

    for (Layer layer in layers) {
      output = layer.forwardPropagation(output ?? inputs);
    }

    // _lastOutput = output;

    return output;
  }

  void backPropagation(List<double> expected) {
    if (!isTraining) runCount++;
    _calculateError(expected);

    // Calculate output layer
    layers[layers.length - 1].backPropagationOutput(expected);

    // calculate input layers
    for (int i = this.layers.length - 2; i >= 0; i--) {
      layers[i].backPropagationHidden(layers[i + 1]);
    }

    // Update all the weights
    this.layers.forEach((l) => l.updateWeights());
  }

  /// Calculate the error of the last output
  void _calculateError(List<double> expected) {
    double expectedSum = 0;
    for (double v in expected) expectedSum += v;
    double calculatedSum = 0;
    for (double v in layers.last.outputs) calculatedSum += v;
    if (expectedSum != 0) {
      double currentError = expectedSum > 0 ? (expectedSum - calculatedSum) / expectedSum * 100 : 0;
      pastErrors.add(currentError);
      // reset average
      averagePercentError = 0;
      // sum up errors
      pastErrors.forEach((e) => averagePercentError += e);
      // divide by length
      averagePercentError /= pastErrors.length;
      if (pastErrors.length > 5 && currentError > 0) {
        pastErrors.removeAt(0);
      }
    }
  }

  /// Removes layer at `removeIndex`
  void removeLayer(int removeIndex) {
    this.wasAltered = true;
    this.layers.removeAt(removeIndex);
  }

  /// Inserts `layer` into `layers` at `index`
  void insetLayer(int index, Layer layer) {
    this.wasAltered = true;
    this.layers.insert(index, layer);
  }

  /// Modifies layer at `updateIndex` to have `newSize` neurons
  void resizeLayer(int updateIndex, int newSize) {
    this.wasAltered = true;
    this.layers[updateIndex].resize(newSize);
  }
}
