package cpen502.nerualnetwork;

import org.apache.commons.math3.linear.BlockRealMatrix;

import java.util.Arrays;
import java.util.function.Function;

/**
 * This is a class for each neural layer.
 * Each layer consists of one or multiple neurons.
 */
public class NeuralLayer {

    /**
     * Layer type. One neural network must consist of at least two layers: input and output.
     *      Input: First layer
     *      Hidden: Middle layers
     *      Output: Last layer
     */
    public enum NeuralLayerType {
        Input, Hidden, Output
    }

    final NeuralLayerType type;
    final NeuralNetwork network;
    final int layerIndex;

    final Function<Double, Double> activationFunc;
    final Function<Double, Double> activationDerivativeFunc;

    final double momentum;
    final double learningRate;

    final Neuron[] neurons;
    final int neuronNum;

    /**
     * Construct a neural layer in a neural network.
     * The neurons are constructed from arrays of neuron types, activation functions and their
     * derivatives, momentums, learning rates.
     * @param type This layer's neurons' types
     * @param neuronNum Neuron number in this layer
     * @param activationFunc The activation function used to compute the output.
     * @param activationDerivativeFunc The derivative of the activation function.
     * @param momentum The momentum used to accelerate the training process.
     * @param learningRate The learning rate used when update the weights.
     * @param network The neural network that includes this neural layer.
     * @param layerIndex The index of this neural layer in the neural network.
     * @throws Exception If the lengths of the arrays used to construct neurons are not the same,
     *                   throws an exception.
     */
    public NeuralLayer(NeuralLayerType type, int neuronNum,
                       Function<Double, Double> activationFunc,
                       Function<Double,Double> activationDerivativeFunc,
                       double momentum, double learningRate,
                       NeuralNetwork network, int layerIndex){
        this.activationFunc = activationFunc;
        this.activationDerivativeFunc = activationDerivativeFunc;
        this.momentum = momentum;
        this.learningRate = learningRate;
        this.type = type;
        this.layerIndex = layerIndex;
        this.network = network;
        this.neuronNum = neuronNum;

        if (type != NeuralLayerType.Output) {
            neurons = new Neuron[neuronNum + 1];
        } else {
            neurons = new Neuron[neuronNum];
        }

        for (int i = 0; i < neurons.length; i ++) {
            neurons[i] = new Neuron(i);
        }

        if (type != NeuralLayerType.Output) {
            neurons[neuronNum].lastInput = 1;
            neurons[neuronNum].lastOutput = 1;
        }
    }

    /**
     * Perform a forward propagation on the current layer.
     * @param weight A weight matrix between this layer and the previous layer.
     *               The row dimension should be the current layer neuron #.
     *               The column dimension should be the prev layer neuron #.
     * @param input A input vector which is the output of previous layer's forward propagation.
     * @return The output is a vector with size of neuronNum
     */
    public double[] forwardPropagate(BlockRealMatrix weight, double[] input) {
        double[] output;
        if (this.type != NeuralLayerType.Input) {
            output = Arrays.copyOf(input, input.length + 1);
            output[output.length - 1] = network.bias;
            output = weight.operate(output);
        } else {
            output = input.clone();
        }

        for (int i = 0; i < output.length; i ++) {
            output[i] = neurons[i].forwardPropagate(activationFunc, output[i]);
        }

        return output;
    }

    /**
     * Perform a backward propagation on the current layer.
     * Also update the weight matrix.
     * @param weight A weight matrix between this layer and the next layer.
     *               The row dimension should be the next layer neuron #.
     *               The column dimension should be the current layer neuron #.
     * @param input A input vector of size [nextLayerNeuron#]
     * @return  The output is a vector with size of neuronNum
     */
    public double[] backwardPropagate(BlockRealMatrix weight, BlockRealMatrix weightDelta, double[] input) {
        // sum of higher layers error * weight
        double[] output = input.clone();
        if (this.type != NeuralLayerType.Output) {
            output = weight.transpose().operate(output);
        }

        for (int i = 0; i < neurons.length; i ++) {
            output[i] = neurons[i].backwardPropagate(activationDerivativeFunc, output[i]);
        }

        // Update weight matrix
        if (this.type != NeuralLayerType.Output) {
            for (int i = 0; i < weight.getRowDimension(); i ++) {
                // Delta for neurons
                for (int j = 0; j < weight.getColumnDimension(); j++) {
                    // Note //
                    double delta = learningRate * input[i] * neurons[j].lastOutput + // error
                            momentum * weightDelta.getEntry(i, j); // momentum
                    weight.setEntry(i, j, weight.getEntry(i, j) + delta); // momentum
                    weightDelta.setEntry(i, j, delta);
                }
            }

            // Get rid of bias turn
            output = Arrays.copyOf(output, output.length-1);
        }

        return output;
    }
}
