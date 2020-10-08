package cpen502.nerualnetwork;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import cpen502.nerualnetwork.NeuralLayer.NeuralLayerType;
import org.apache.commons.math3.linear.BlockRealMatrix;
import sun.jvm.hotspot.opto.Block;

/**
 * Neural network class that implements CommonInterface
 * A neural network consists of at least two neural layers, one for the input
 * and one for the output.
 * This class also contains some functions that delivers the stats of this neural
 * network and its trainings.
 */
public class NeuralNetwork{
    final double bias = 1.0;

    final static int MIN_NEURAL_LAYER_NUM = 2;
    final NeuralLayer[] layers;
    final int layerNum;

    final BlockRealMatrix[] weights;
    final BlockRealMatrix[] weightDeltas;

    private final Random random = new Random();

    public NeuralNetwork (int[] neuronNums, Function<Double, Double>[] activationFuncs,
                          Function<Double, Double>[] activationDerivativeFuncs,
                          double[] momentums, double[] learningRates) throws Exception {
        // Validate parameters
        layerNum = neuronNums.length;
        if (activationFuncs.length != layerNum || activationDerivativeFuncs.length != layerNum ||
            momentums.length != layerNum || learningRates.length != layerNum) {
            throw new Exception("Lists of parameters don't agree on sizes");
        }

        if (layerNum < MIN_NEURAL_LAYER_NUM) {
            throw new Exception("There should be at least 2 layers in a neural network");
        }

        layers = new NeuralLayer[layerNum];

        // Construct neural layers
        // First layer is input layer
        layers[0] = new NeuralLayer(NeuralLayerType.Input, neuronNums[0],
                activationFuncs[0], activationDerivativeFuncs[0],
                momentums[0], learningRates[0], this, 0);
        // Last layer is output layer
        layers[layerNum-1] = new NeuralLayer(NeuralLayerType.Output, neuronNums[layerNum-1],
                activationFuncs[layerNum-1], activationDerivativeFuncs[layerNum-1],
                momentums[layerNum-1], learningRates[layerNum-1], this, 0);
        for (int i = 1; i < neuronNums.length - 1; i ++) {
                layers[i] = new NeuralLayer(NeuralLayerType.Hidden, neuronNums[i],
                        activationFuncs[i], activationDerivativeFuncs[i],
                        momentums[i], learningRates[i], this, i);
        }

        // Construct weights and biases
        // Weights are matrices between each layers
        weights = new BlockRealMatrix[layerNum-1];
        weightDeltas = new BlockRealMatrix[layerNum-1];

        for (int i = 0; i < weights.length; i ++) {
            weights[i] = new BlockRealMatrix(layers[i+1].neuronNum, layers[i].neuronNum + 1);
            weightDeltas[i] = new BlockRealMatrix(layers[i+1].neuronNum, layers[i].neuronNum + 1);
        }
    }

    /**
     * Learn the expected value based on the input vector.
     * This is what it does:
     *      1. Compute the output using the current weights.
     *      2. Update parameters (weight, step size, etc.) in each neuron.
     * @param X The input vector. The length should match the layer 1 size.
     * @param argValue The new value to learn
     * @return The output using the neural network before it updated
     */
    public double[] train(double[] X, double[] argValue) {
        double[] output = forwardPropagation(X);

        double[] input = new double[argValue.length];
        for (int i = 0; i < input.length; i ++) {
            input[i] = argValue[i] - output[i];
        }
        backwardPropagation(input);

        return output;
    }

    /**
     * Compute the output based on the input vector.
     * This method will not update the neural network.
     * @param X The input vector. The length should match the layer 1 size.
     * @return the value computed using the current network and the input vector
     */
    public double[] outputFor(double[] X) {
        double[] output = forwardPropagation(X);
        return output;
    }

    /**
     * Perform a forward propagation from a matrix of input. The output is generated at the
     * end of the neural network;
     * @param input A vector of size [firstLayerNeuron#]
     * @return A vector from last layer of size [lastLayerNeuron#]
     */
    private double[] forwardPropagation(double[] input) {
        double[] output = layers[0].forwardPropagate(null, input);

        for (int i = 0; i < weights.length; i ++) {
            output = layers[i+1].forwardPropagate(weights[i], output);
        }

        return output;
    }

    /**
     * Perform a backward propagation from a matrix of errors.
     * Also update the weights on the run.
     * @param input A vector of errors between the expected output and the actual output.
     *              Each element of the vector is (C_i - y_i), where C_i is the expected output,
     *              and y_i is the actual output.
     */
    private void backwardPropagation(double[] input) {
        double[] output = layers[layerNum-1].backwardPropagate(null, null, input);

        for (int i = weights.length - 1; i >= 0; i --) {
            output = layers[i].backwardPropagate(weights[i], weightDeltas[i], output);
        }
    }


    public void initializeWeights() {
        for (int i = 0; i < weights.length; i ++) {
            for (int row = 0; row < weights[i].getRowDimension(); row ++) {
                for (int column = 0; column < weights[i].getColumnDimension(); column ++) {
                    weights[i].setEntry(row, column, random.nextDouble() - 0.5);
                    weightDeltas[i].setEntry(row, column, 0);
                }
            }
        }
    }

    public void zeroWeights() {
        for (int i = 0; i < weights.length; i ++) {
            for (int row = 0; row < weights[i].getRowDimension(); row ++) {
                for (int column = 0; column < weights[i].getColumnDimension(); column ++) {
                    weights[i].setEntry(row, column, 0);
                    weightDeltas[i].setEntry(row, column, 0);
                }
            }
        }
    }


    /**
     * Save the current configurations of this neural network.
     * @param argFile of type File.
     */
    public void save(File argFile) {

    }

    /**
     * Reconstruct a neural network based on the states read from a file.
     * @param argFileName
     * @throws IOException
     */
    public void load(String argFileName) throws IOException {

    }
}
