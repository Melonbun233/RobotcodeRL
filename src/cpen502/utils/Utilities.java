package cpen502.utils;

import cpen502.nerualnetwork.Neuron;

import java.util.ArrayList;
import java.util.List;

public class Utilities {
    public static List<Neuron[]> generateNeuronList(int[] neuronNums) {
        List<Neuron[]> list = new ArrayList<>();
        for (int neuronNum : neuronNums) {
            Neuron[] neurons = new Neuron[neuronNum];
            for (int i = 0; i < neuronNum; i ++) {
                neurons[i] = new Neuron(i);
            }
            list.add(neurons);
        }
        return list;
    }

    public static double calculateTotalError(double[][] expected, double[][] actual) {
        double error = 0.0;
        for (int i = 0; i < expected.length; i ++) {
            for (int j = 0; j < expected[0].length; j ++) {
                error += Math.pow(actual[i][j] - expected[i][j], 2);
            }
        }

        return 0.5*error;
    }
}
