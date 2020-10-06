package cpen502.nerualnetwork;

import org.apache.commons.math3.linear.BlockRealMatrix;

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
    double bias;

    final Neuron[] neurons;
    final int neuronNum;

    /**
     * Construct a neural layer in a neural network.
     * The neurons are constructed from arrays of neuron types, activation functions and their
     * derivatives, momentums, learning rates.
     * @param type This layer's neurons' types
     * @param neurons All neurons in this layer.
     * @param activationFunc The activation function used to compute the output.
     * @param activationDerivativeFunc The derivative of the activation function.
     * @param momentum The momentum used to accelerate the training process.
     * @param learningRate The learning rate used when update the weights.
     * @param network The neural network that includes this neural layer.
     * @param layerIndex The index of this neural layer in the neural network.
     * @throws Exception If the lengths of the arrays used to construct neurons are not the same,
     *                   throws an exception.
     */
    public NeuralLayer(NeuralLayerType type, Neuron[] neurons,
                       Function<Double, Double> activationFunc,
                       Function<Double,Double> activationDerivativeFunc,
                       double momentum, double learningRate, double bias,
                       NeuralNetwork network, int layerIndex){
        this.neurons = neurons;
        this.activationFunc = activationFunc;
        this.activationDerivativeFunc = activationDerivativeFunc;
        this.momentum = momentum;
        this.learningRate = learningRate;
        this.bias = bias;
        this.neuronNum = neurons.length;
        this.type = type;
        this.layerIndex = layerIndex;
        this.network = network;
    }

    /**
     * Perform a forward propagation on the current layer.
     * @param weight A weight matrix between this layer and the previous layer.
     *               The row dimension should be the current layer neuron #.
     *               The column dimension should be the prev layer neuron #.
     * @param input A input vector which is the output of previous layer's forward propagation.
     * @return The output is a vector with size of neuronNum
     */
    public double[] forwardPropagate(BlockRealMatrix weight, double[] input, double[] biasWeight) {
        double[] output = input.clone();
        output = this.type == NeuralLayerType.Input ? output : weight.operate(output);

        // Sum of lower levels output * weight
        for (int i = 0; i < output.length; i ++) {
            output[i] = neurons[i].forwardPropagate(activationFunc,
                    this.type == NeuralLayerType.Input ? output[i] : output[i] + bias * biasWeight[i]);
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
    public double[] backwardPropagate(BlockRealMatrix weight, BlockRealMatrix weightDelta,
                                      double[] input, double[] biasWeight, double[] biasWeightDelta) {
        // Calculate sum of errors
        double[] output = input.clone();
        output = this.type == NeuralLayerType.Output ? output : weight.transpose().operate(output);

        // sum of higher layers error * weight
        for (int i = 0; i < output.length; i ++) {
            output[i] = neurons[i].backwardPropagate(activationDerivativeFunc, output[i]);
        }

        // Update weight matrix delta
        if (this.type != NeuralLayerType.Output) {
            for (int i = 0; i < weight.getRowDimension(); i ++) {
                double delta = momentum * biasWeightDelta[i] + learningRate * input[i];
                biasWeight[i] += delta;
                biasWeightDelta[i] = delta;
                for (int j = 0; j < weight.getColumnDimension(); j++) {
                    // Note //
                    delta = learningRate * input[i] * neurons[j].lastOutput + // error
                            momentum * weightDelta.getEntry(i, j); // momentum
                    weight.setEntry(i, j, weight.getEntry(i, j) + delta); // momentum
                    weightDelta.setEntry(i, j, delta);
                }
            }
        }

        return output;
    }
}
