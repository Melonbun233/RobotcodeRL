package cpen502.nerualnetwork;

import java.io.File;
import java.io.IOException;

/**
 * Neural network class that implements CommonInterface
 * A neural network consists of at least two neural layers, one for the input
 * and one for the output.
 * This class also contains some functions that delivers the stats of this neural
 * network and its trainings.
 */
public class NeuralNetwork implements CommonInterface {

    public NeuralNetwork () {

    }

    /**
     * Compute the output based on the input vector.
     * This method will not update the neural network.
     * @param X The input vector. The length should match the layer 1 size.
     * @return the value computed using the current network and the input vector
     */
    @Override
    public double outputFor(double[] X) {
        return 0;
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
    @Override
    public double train(double[] X, double argValue) {
        return 0;
    }

    /**
     * Save the current configurations of this neural network.
     * @param argFile of type File.
     */
    @Override
    public void save(File argFile) {

    }

    /**
     * Reconstruct a neural network based on the states read from a file.
     * @param argFileName
     * @throws IOException
     */
    @Override
    public void load(String argFileName) throws IOException {

    }
}
