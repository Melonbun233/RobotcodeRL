package cpen502.nerualnetwork;

import java.util.function.Function;

/**
 * This is a class for each neural layer.
 * Each layer consists of one or multiple neurons.
 */
public class NeuralLayer {
    final NeuralNetwork network;
    final int layerIndex;

    final Neuron[] neurons;
    final int neuronNum;

    /**
     * Construct a neural layer in a neural network.
     * The neurons are constructed from arrays of neuron types, activation functions and their
     * derivatives, momentums, learning rates.
     * @param neuronTypes This layer's neurons' types.
     * @param activationFuncs This layer's neurons' activation functions.
     * @param activationDerivativeFuncs This layer's neurons' activation functions' derivatives.
     * @param momentums This layer's neurons' momentums.
     * @param learningRates This layer's neurons' learning rates.
     * @param network The neural network that includes this neural layer.
     * @param layerIndex The index of this neural layer in the neural network.
     * @throws Exception If the lengths of the arrays used to construct neurons are not the same,
     *                   throws an exception.
     */
    public NeuralLayer(NeuronType[] neuronTypes, Function<Double, Double>[] activationFuncs,
                       Function<Double, Double>[] activationDerivativeFuncs,
                       double[] momentums, double[] learningRates, NeuralNetwork network,
                       int layerIndex) throws Exception{
        // Validate the input parameters
        neuronNum = neuronTypes.length;
        if (activationFuncs.length != neuronNum || activationDerivativeFuncs.length != neuronNum ||
            momentums.length != neuronNum || learningRates.length != neuronNum) {
            throw new Exception("NeuralLayer Constructor: Some arrays' length are not the same. " +
                    "Failed to construct a neural layer");
        }
        // Construct the neurons in this layer
        neurons = new Neuron[neuronNum];
        for (int i = 0; i < neuronNum; i ++) {
            neurons[i] = new Neuron(neuronTypes[i], activationFuncs[i], activationDerivativeFuncs[i],
                    momentums[i], learningRates[i], i, this);
        }

        this.layerIndex = layerIndex;
        this.network = network;
    }
}
