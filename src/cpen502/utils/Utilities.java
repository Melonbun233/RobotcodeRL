package cpen502.utils;


public class Utilities {
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
