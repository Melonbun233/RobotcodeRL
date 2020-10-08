package cpen502.nerualnetwork;

import cpen502.utils.Functions;
import cpen502.utils.Utilities;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;


class NeuralNetworkTest {
    public static void main(String[] args){
        int[] neuronNums = new int[]{2, 4, 1};
        Function<Double, Double>[] activationFuncs = new Function[3];
        Arrays.fill(activationFuncs, Functions.sigmoidBinary);

        Function<Double, Double>[] activationDerivativeFuncs = new Function[3];
        Arrays.fill(activationDerivativeFuncs, Functions.sigmoidDerivativeBinary);

        double[] momentums = new double[] {0.9 , 0.9, 0.9};
        double[] learningRates = new double[] {0.02, 0.02, 0.02};

        // Initialize neural network
        NeuralNetwork neuralnet;
        try {
            neuralnet = new NeuralNetwork(neuronNums, activationFuncs, activationDerivativeFuncs,
                    momentums, learningRates);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        for (int i = 0; i < 4; i ++) {
            neuralnet.weights[1].setEntry(0, i, i + 2);
            for (int j = 0; j < 2; j ++) {
                neuralnet.weights[0].setEntry(i, j, i*3+j+2);
            }
        }

        neuralnet.outputFor(new double[]{1,1});
    }
}