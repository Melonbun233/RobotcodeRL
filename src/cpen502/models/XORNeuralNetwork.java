package cpen502.models;

import cpen502.nerualnetwork.NeuralNetwork;
import cpen502.nerualnetwork.Neuron;
import cpen502.utils.Functions;
import cpen502.utils.Utilities;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class XORNeuralNetwork {

    public static double[][] getTrainingSetBinary() {
        return new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    }

    public static double[][] getAnswerSetBinary() {
        return new double[][]{{0}, {1}, {1}, {0}};
    }

    public static double[][] getTrainingSetBipolar() {
        return new double[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    }

    public static double[][] getAnswerSetBipolar() {
        return new double[][]{{-1}, {1}, {1}, {-1}};
    }

    public static void main(String[] args) {
        boolean useBipolar = true;
        int runNum = 10;

        List<Neuron[]> neuronList = Utilities.generateNeuronList(new int[]{2, 4, 1});

        Function<Double, Double>[] activationFuncs = new Function[3];
        Arrays.fill(activationFuncs, useBipolar ?
                Functions.sigmoidBipolar : Functions.sigmoidBinary);

        Function<Double, Double>[] activationDerivativeFuncs = new Function[3];
        Arrays.fill(activationDerivativeFuncs, useBipolar ?
                Functions.sigmoidDerivativeBipolar : Functions.sigmoidDerivativeBinary);

        double[] momentums = new double[] {0.9 , 0.9, 0.9};
        double[] learningRates = new double[] {0.02, 0.02, 0.02};

        // Initialize neural network
        NeuralNetwork neuralnet;
        try {
            neuralnet = new NeuralNetwork(neuronList, activationFuncs, activationDerivativeFuncs,
                    momentums, learningRates);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        // Train the neural network
        double[][] trainingSet = useBipolar ?
                XORNeuralNetwork.getTrainingSetBipolar() : XORNeuralNetwork.getTrainingSetBinary();
        double[][] answerSet = useBipolar ?
                XORNeuralNetwork.getAnswerSetBipolar() : XORNeuralNetwork.getAnswerSetBinary();
        double[][] output = new double[answerSet.length][answerSet[0].length];

        int totalEpochNum = 0;
        int epochNum;
        double totalError;
        String useBipolarStr = useBipolar ? "-bipolar" : "-binary";

        for (int run = 0; run < runNum; run ++) {

            Writer writer = null;
            try {
                writer = new BufferedWriter(new OutputStreamWriter(
                        new FileOutputStream("trail-" + run + useBipolarStr + ".txt"),
                        "utf-8"));

                neuralnet.initializeWeights();
                epochNum = 0;
                totalError = Double.POSITIVE_INFINITY;

                while (totalError > 0.05) {
                    // Train one epoch
                    for (int i = 0; i < trainingSet.length; i++) {
                        neuralnet.train(trainingSet[i], answerSet[i]);
                    }

                    for (int i = 0; i < trainingSet.length; i++) {
                        output[i] = neuralnet.outputFor(trainingSet[i]);
                    }

                    totalError = Utilities.calculateTotalError(answerSet, output);
                    epochNum++;
                    totalEpochNum++;

                    if (epochNum > 1000000) {
                        System.out.println("Exceeding max epoch nums");
                        totalEpochNum -= 1000000;
                        run--;
                        break;
                    }
                    writer.write(epochNum + "," + totalError + "\n");
                    // System.out.println("Total Error of Epoch " + epochNum + ": " + totalError);
                }
                //writer.write("Run " + run + " ended with " + epochNum + "\n");
                System.out.println("Run " + run + " ended with " + epochNum);

                writer.close();
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
                break;
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                break;
            } catch (IOException e) {
                e.printStackTrace();
                break;
            }

        }

        System.out.println("Avg epoch number needed to converge: " + totalEpochNum / runNum);
    }
}
