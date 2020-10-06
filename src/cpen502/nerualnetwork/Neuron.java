package cpen502.nerualnetwork;

import java.util.function.Function;

/**
 * This is a class for each Neuron.
 * The Neuron is the smallest component in a neural network.
 */
public class Neuron {
    final int neuronIndex; // Position index in this layer.

    double lastInput;
    double lastOutput;

    /**
     * Construct a neuron in a neural layer.
     * @param neuronIndex The index of this neuron's position in the neural layer.
     */
    public Neuron(int neuronIndex) {
        this.neuronIndex = neuronIndex;
    }

    /**
     * This function should be called during the forward propagation process.
     * Compute the output based on the input value using the activation function.
     * Also store the output in this neuron.
     * @param func The activation function.
     * @param input The input is calculated as: dotProduct(prevLayerWeights, preLayerOutputs)
     * @return The computed output.
     */
    public double forwardPropagate(Function<Double, Double> func, double input) {
        lastInput = input;
        lastOutput = func.apply(input);
        return lastOutput;
    }

    /**
     * This function should be called during the backward propagation process.
     * Compute the error based on the weighted error sum of the next layer.
     * Also store the error in this neuron.
     * @param func The derivative of the activation function used on this neuron.
     * @param input The error from the next layer.
     *              If this neuron is at output layer:
     *                  input = expectedOutput - actualOutput
     *              Otherwise:
     *                  input = sum(nextLayerWeights .* nextLayerErrors)
     * @return The computed error.
     */
    public double backwardPropagate(Function<Double, Double> func, double input) {
        return input * func.apply(lastOutput);
    }

}
