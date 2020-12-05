package cpen502.nerualnetwork;

import javax.annotation.processing.Filer;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

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
    final Random random = new Random();

    final int layerNum;
    final int[] neuronNums;
    final List<double[][]> weights;
    final List<double[][]> weightDeltas;
    final List<double[][]> weightCorrections;

    final List<double[]> neuronOutputs;
    final List<double[]> neuronDeltas;

    final double momentum;
    final double learningRate;

    final Function<Double, Double> activationFunction;
    final Function<Double, Double> activationDerivativeFunction;

    public NeuralNetwork (int[] neuronNums, Function<Double, Double> activationFunction,
                          Function<Double, Double> activationDerivativeFunction,
                          double momentum, double learningRate) throws Exception {
        if (neuronNums.length < MIN_NEURAL_LAYER_NUM) {
            throw new Exception("Neural Network must have at least " + MIN_NEURAL_LAYER_NUM + " layers");
        }
        layerNum = neuronNums.length;
        this.neuronNums = neuronNums;
        this.activationFunction = activationFunction;
        this.activationDerivativeFunction = activationDerivativeFunction;
        this.momentum = momentum;
        this.learningRate = learningRate;

        // create all weights
        this.weights = new ArrayList<>();
        this.weightDeltas = new ArrayList<>();
        this.weightCorrections = new ArrayList<>();
        for (int i = 0; i < layerNum - 1; i ++) {
            this.weights.add(new double [neuronNums[i] + 1][neuronNums[i+1]]);
            this.weightDeltas.add(new double [neuronNums[i] + 1][neuronNums[i+1]]);
            this.weightCorrections.add(new double [neuronNums[i] + 1][neuronNums[i+1]]);
        }

        // cache to remember the last output
        this.neuronOutputs = new ArrayList<>();
        this.neuronDeltas = new ArrayList<>();
        for (int i = 0; i < layerNum; i ++) {
            this.neuronOutputs.add(new double[neuronNums[i]]);
            this.neuronDeltas.add(new double[neuronNums[i]]);
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

        double[] input = new double[output.length];
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
    double[] forwardPropagation(double[] input) {
        // process through the input layer
        double[] output = input.clone();
        for (int i = 0; i < input.length; i ++) {
            // output[i] = activationFunction.apply(input[i]);
            neuronOutputs.get(0)[i] = output[i];
        }


        // process through the hidden layers and output layers
        for (int i = 1; i < layerNum; i ++) {
            double[][] weight = weights.get(i-1);
            double[] neuronOutput = neuronOutputs.get(i);

            double[] nextOutput = new double[neuronNums[i]];

            for (int outputIndex = 0; outputIndex < neuronNums[i]; outputIndex ++) {
                // weight sum
                for (int inputIndex = 0; inputIndex < weight.length; inputIndex ++) {
                    // bias neuron
                    if (inputIndex == weight.length - 1){
                        nextOutput[outputIndex] += weight[inputIndex][outputIndex] * bias;
                    } else {
                        nextOutput[outputIndex] += weight[inputIndex][outputIndex] * output[inputIndex];
                    }
                }
                // activate
                nextOutput[outputIndex] = activationFunction.apply(nextOutput[outputIndex]);
                // cache output
                neuronOutput[outputIndex] = nextOutput[outputIndex];
            }

            output = nextOutput;
        }

        // output
        return output;
    }

    /**
     * Perform a backward propagation from a matrix of errors.
     * Also update the weights on the run.
     * @param input A vector of errors between the expected output and the actual output.
     *              Each element of the vector is (C_i - y_i), where C_i is the expected output,
     *              and y_i is the actual output.
     */
    void backwardPropagation(double[] input) {
        // process through the output layer and hidden layers
        for (int i = layerNum - 1; i >= 1; i --) {
            double[][] weight = i == layerNum - 1 ? null : weights.get(i); // use weight above this layer
            double[][] weightCorrection = weightCorrections.get(i-1); // correct weight below this layer
            double[] neuronDelta = neuronDeltas.get(i);
            double[] nextNeuronDelta = i == layerNum - 1 ? null : neuronDeltas.get(i+1);
            double[] neuronOutput = neuronOutputs.get(i);
            double[] prevNeuronOutput = neuronOutputs.get(i-1);

            for (int neuronIndex = 0; neuronIndex < neuronNums[i]; neuronIndex ++) {
                // calculate deltas for this layer's neurons
                neuronDelta[neuronIndex] = 0;
                if (i == layerNum - 1) {
                    // output layer delta
                    neuronDelta[neuronIndex] = input[neuronIndex] *
                            activationDerivativeFunction.apply(neuronOutput[neuronIndex]);
                } else {
                    // hidden layers, sum the weighted next layer delta
                    for (int nextNeuronIndex = 0; nextNeuronIndex < neuronNums[i+1]; nextNeuronIndex ++) {
                        neuronDelta[neuronIndex] += nextNeuronDelta[nextNeuronIndex] *
                                weight[neuronIndex][nextNeuronIndex];
                    }
                    neuronDelta[neuronIndex] *= activationDerivativeFunction.apply(neuronOutput[neuronIndex]);
                }

                // calculate weight correction terms
                for (int weightIndex = 0; weightIndex < weightCorrection.length; weightIndex ++) {
                    // bias neuron
                    if (weightIndex == weightCorrection.length - 1) {
                        weightCorrection[weightIndex][neuronIndex] = learningRate * neuronDelta[neuronIndex];
                    } else {
                        weightCorrection[weightIndex][neuronIndex] =
                                learningRate * neuronDelta[neuronIndex] * prevNeuronOutput[weightIndex];
                    }
                }
            }
//            // try update weight here
//            weight = weights.get(i-1);
//            double[][] weightDelta = weightDeltas.get(i-1);
//            for (int row = 0; row < weight.length; row ++) {
//                for (int column = 0; column < weight[0].length; column ++) {
//                    double delta = weightCorrection[row][column] + momentum * weightDelta[row][column];
//                    weight[row][column] += delta;
//                    weightDelta[row][column] = delta;
//                }
//            }
        }

        // update weights
        for (int i = 0; i < weights.size(); i ++) {
            double[][] weight = weights.get(i);
            double[][] weightDelta = weightDeltas.get(i);
            double[][] weightCorrection = weightCorrections.get(i);
            for (int row = 0; row < weight.length; row ++) {
                for (int column = 0; column < weight[0].length; column ++) {
                    double delta = weightCorrection[row][column] + momentum * weightDelta[row][column];
                    weight[row][column] += delta;
                    weightDelta[row][column] = delta;
                    weightCorrection[row][column] = 0;
                }
            }
        }
    }


    public void initializeWeights() {
        for (int i = 0; i < weights.size(); i ++) {
            double[][] weight = weights.get(i);
            double[][] weightDelta = weightDeltas.get(i);
            for (int row = 0; row < weight.length; row ++) {
                for (int column = 0; column < weight[0].length; column ++) {
                    weight[row][column] = random.nextDouble() - 0.5;
                    weightDelta[row][column] = 0;
                }
            }
        }
    }

    public void zeroWeights() {
        for (int i = 0; i < weights.size(); i ++) {
            double[][] weight = weights.get(i);
            double[][] weightDelta = weightDeltas.get(i);
            for (int row = 0; row < weight.length; row ++) {
                for (int column = 0; column < weight[0].length; column ++) {
                    weight[row][column] = 0;
                    weightDelta[row][column] = 0;
                }
            }
        }
    }


    public void save(File argFile) {
        PrintStream ps = null;
        try {
            ps = new PrintStream((new FileOutputStream(argFile)));
            for (int i = 0; i < weights.size(); i ++) {
                for (int j = 0; j < weights.get(i).length; j ++) {
                    for (int k = 0; k < weights.get(i)[j].length; k ++) {
                        ps.println(weights.get(i)[j][k]);
                        ps.println(weightDeltas.get(i)[j][k]);
                        ps.println(weightCorrections.get(i)[j][k]);
                    }
                }
            }

            for (int i = 0; i < neuronOutputs.size(); i ++) {
                for (int j = 0; j < neuronOutputs.get(i).length; j ++) {
                    ps.println(neuronOutputs.get(i)[j]);
                    ps.println(neuronDeltas.get(i)[j]);
                }
            }
            ps.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void load(File argFile)  {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(argFile));
            for (int i = 0; i < weights.size(); i ++) {
                for (int j = 0; j < weights.get(i).length; j ++) {
                    for (int k = 0; k < weights.get(i)[j].length; k ++) {
                        weights.get(i)[j][k] = Double.parseDouble(reader.readLine());
                        weightDeltas.get(i)[j][k] = Double.parseDouble(reader.readLine());
                        weightCorrections.get(i)[j][k] = Double.parseDouble(reader.readLine());
                    }
                }
            }

            for (int i = 0; i < neuronOutputs.size(); i ++) {
                for (int j = 0; j < neuronOutputs.get(i).length; j ++) {
                    neuronOutputs.get(i)[j] = Double.parseDouble(reader.readLine());
                    neuronDeltas.get(i)[j] = Double.parseDouble(reader.readLine());
                }
            }

            reader.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
