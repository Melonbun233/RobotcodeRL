package cpen502.nerualnetwork;

import java.util.function.Function;

/**
 * This is a class for each Neuron.
 * The Neuron is the smallest component in a neural network.
 */
public class Neuron {
    final NeuralLayer parentLayer; // The neural layer that has this neuron.
    final int neuronIndex; // Position index in this layer.
    final NeuronType type; // Input, Output, or Hidden.

    final Function<Double, Double> activationFunc;
    final Function<Double, Double> activationDerivativeFunc;

    double momentum;
    double learningRate;

    double lastInput;
    double lastOutput;
    double lastError;

    /**
     * Construct a neuron in a neural layer.
     * @param type The type of this nuron.
     * @param activationFunc The activation function used to compute the output.
     * @param activationDerivativeFunc The derivative of the activation function.
     * @param momentum The momentum used to accelerate the training process.
     * @param learningRate The learning rate used when update the weights.
     * @param neuronIndex The index of this neuron's position in the neural layer.
     * @param parentLayer The neural layer that includes this neuron.
     */
    public Neuron(NeuronType type, Function<Double, Double> activationFunc,
                  Function<Double, Double> activationDerivativeFunc, double momentum,
                  double learningRate, int neuronIndex, NeuralLayer parentLayer) {
        this.type = type;
        this.activationFunc = activationFunc;
        this.activationDerivativeFunc = activationDerivativeFunc;
        this.momentum = momentum;
        this.learningRate = learningRate;
        this.neuronIndex = neuronIndex;
        this.parentLayer = parentLayer;
    }

    /**
     * This function should be called during the forward propagation process.
     * Compute the output based on the input value using the activation function.
     * Also store the output in this neuron.
     * @param input The input is calculated as: dotProduct(prevLayerWeights, preLayerOutputs)
     * @return The computed output.
     */
    public double activate(double input) {
        lastInput = input;
        lastOutput = activationFunc.apply(input);
        return lastOutput;
    }

    /**
     * This function should be called during the backward propagation process.
     * Compute the error based on the weighted error sum of the next layer.
     * Also store the error in this neuron.
     * @param weightedErrorSum The weighted error sum of the next layer. It is computed as:
     *                         sum(nextLayerWeights .* nextLayerErrors)
     * @return The computed error.
     */
    public double updateError(double weightedErrorSum) {
        lastError = weightedErrorSum * activationDerivativeFunc.apply(lastOutput);
        return lastError;
    }

}
