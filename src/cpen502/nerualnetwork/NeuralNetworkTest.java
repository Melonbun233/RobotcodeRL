package cpen502.nerualnetwork;

import cpen502.utils.Functions;

import java.util.Arrays;
import java.util.function.Function;

public class NeuralNetworkTest {

    public static void main(String[] args) {
        int[] nueronNums = new int[]{2, 4, 1};

        Function<Double, Double> activationFunction = Functions.sigmoidBinary;
        Function<Double, Double> activationDerivativeFunction = Functions.sigmoidDerivativeBinary;

        double momentums = 0;
        double learningRates = 0.02;

        // Initialize neural network
        NeuralNetwork neuralnet;
        try {
            neuralnet = new NeuralNetwork(nueronNums, activationFunction,
                    activationDerivativeFunction, momentums, learningRates);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 4; j ++) {
                neuralnet.weights.get(0)[i][j] = j + 1;
            }
            Arrays.fill(neuralnet.neuronOutputs.get(i), 1);
        }

        for (int i = 0; i < 5; i ++) {
            neuralnet.weights.get(1)[i][0] = i + 1;
        }


        neuralnet.backwardPropagation(new double[]{1});

    }
}
