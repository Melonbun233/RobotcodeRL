package cpen502.models;

import cpen502.nerualnetwork.NeuralNetwork;
import cpen502.robots.QLearningRobot;
import cpen502.utils.Functions;

import java.io.*;
import java.util.Map;
import java.util.function.Function;

public class RobocodeNN {
    // some parameters for lut dimensions
    public final static int lutDepth = QLearningRobot.stateNum;
    public final static int actionDim = QLearningRobot.actionNum;

    public final static Map<QLearningRobot.StateCategory, Integer> stateDim = QLearningRobot.stateDim;

    public final static int posXDim = stateDim.get(QLearningRobot.StateCategory.PosX);
    public final static int posYDim = stateDim.get(QLearningRobot.StateCategory.PosY);
    public final static int energyDim = stateDim.get(QLearningRobot.StateCategory.Energy);
    public final static int enemyDistanceDim = stateDim.get(QLearningRobot.StateCategory.EnemyDistance);
    public final static int gunHeatDim = stateDim.get(QLearningRobot.StateCategory.GunHeated);


    static double[][][][][][] lut = new double[posXDim][posYDim][energyDim]
            [enemyDistanceDim][gunHeatDim][actionDim];
    static String lutFilename = "result/assignment3/LUT.txt";

    public static void main(String[] args) throws IOException {
        // Load the training set
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(lutFilename)));
            for (int a = 0; a < posXDim; a ++) {
                for (int b = 0; b < posYDim; b ++) {
                    for (int c = 0; c < energyDim; c ++) {
                        for (int d = 0; d < enemyDistanceDim; d ++) {
                            for (int e = 0; e < gunHeatDim; e ++) {
                                for (int f = 0; f < actionDim; f ++) {
                                    lut[a][b][c][d][e][f] = Double.parseDouble(reader.readLine());
                                }
                            }
                        }
                    }
                }
            }
            reader.close();
        } catch (FileNotFoundException fileNotFoundException) {
            fileNotFoundException.printStackTrace();
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }

        // normalize training set output values
        double maxQ = Double.NEGATIVE_INFINITY;
        double minQ = Double.POSITIVE_INFINITY;
        for (int a = 0; a < posXDim; a ++) {
            for (int b = 0; b < posYDim; b ++) {
                for (int c = 0; c < energyDim; c ++) {
                    for (int d = 0; d < enemyDistanceDim; d ++) {
                        for (int e = 0; e < gunHeatDim; e ++) {
                            for (int f = 0; f < actionDim; f ++) {
                                if (maxQ < lut[a][b][c][d][e][f]) {
                                    maxQ = lut[a][b][c][d][e][f];
                                }
                                if (minQ > lut[a][b][c][d][e][f]) {
                                    minQ = lut[a][b][c][d][e][f];
                                }
                            }
                        }
                    }
                }
            }
        }

        Function<Double, Double> activationFunction = Functions.sigmoidBipolar;
        Function<Double, Double> activationDerivativeFunction = Functions.sigmoidDerivativeBipolar;
        int hiddenNeuronNum = 80;
        double momentums = 0.6;
        double learningRates = 0.01;
        int[] neuronNums = new int[] {20, hiddenNeuronNum, 1};
        NeuralNetwork neuralnet;
        try {
            neuralnet = new NeuralNetwork(neuronNums, activationFunction,
                    activationDerivativeFunction, momentums, learningRates);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        int runNum = 1;
        int totalEpochNum = 0;
        int epochNum;
        int maxEpochNum = 10000;
        double totalError;

        boolean loadFile = false;
        String loadFileName = "result/assignment3/NN.txt";
        boolean saveFile = true;
        String saveFileName = "result/assignment3/NN.txt";
        boolean saveLog = false;

        for (int run = 0; run < runNum; run ++) {
            neuralnet.initializeWeights();
            if (loadFile) neuralnet.load(new File(loadFileName));
            epochNum = 0;
            totalError = Double.POSITIVE_INFINITY;

            Writer writer = null;
            if (saveLog) {
                writer = new BufferedWriter(new OutputStreamWriter(
                        new FileOutputStream("result/assignment3/NN-" + hiddenNeuronNum
                                + "/" + momentums + "-" + learningRates + "-trial-" + run + ".txt"),
                        "utf-8"
                ));
            }

            while (epochNum < maxEpochNum) {
                totalError = 0;
                // Train 1 epoch
                for (int a = 0; a < posXDim; a ++) {
                    for (int b = 0; b < posYDim; b ++) {
                        for (int c = 0; c < energyDim; c ++) {
                            for (int d = 0; d < enemyDistanceDim; d ++) {
                                for (int e = 0; e < gunHeatDim; e ++) {
                                    for (int f = 0; f < actionDim; f ++) {
                                        double[] sa = SAToOneHot(a, b, c, d, e, f);
                                        double[] q = new double[1];
                                        q[0] = normalizeQ(maxQ, minQ,
                                                lut[a][b][c][d][e][f]);
                                        neuralnet.train(sa, q);
                                    }
                                }
                            }
                        }
                    }
                }

                for (int a = 0; a < posXDim; a ++) {
                    for (int b = 0; b < posYDim; b ++) {
                        for (int c = 0; c < energyDim; c ++) {
                            for (int d = 0; d < enemyDistanceDim; d ++) {
                                for (int e = 0; e < gunHeatDim; e ++) {
                                    for (int f = 0; f < actionDim; f ++) {
                                        double[] sa = SAToOneHot(a, b, c, d, e, f);
                                        double output = neuralnet.outputFor(sa)[0];
                                        double q = normalizeQ(maxQ, minQ, lut[a][b][c][d][e][f]);
                                        double error = Math.pow(output - q, 2) / 2.0;
                                        totalError += error;
                                    }
                                }
                            }
                        }
                    }
                }

                epochNum ++;
                totalEpochNum ++;
                if (saveLog) writer.write(epochNum + "," + totalError + "\n");
                System.out.println("Total Error of Epoch " + epochNum + ": " + totalError);
            }

            System.out.println("Run " + run + " ended with " + epochNum);
            System.out.println("Last error " + totalError);
            if (saveLog) writer.close();
        }
        if (saveFile) neuralnet.save(new File(saveFileName));
        System.out.println("Avg epoch number needed to converge: " + totalEpochNum / runNum);
    }

    static double normalizeQ(double maxQ, double minQ, double Q) {
        double halfSpace = (maxQ - minQ) / 2.0;
        double mid = (maxQ + minQ) / 2.0;
        return (Q - mid)/halfSpace;
    }

    static double[] SAToOneHot(int posX, int posY, int energy, int enemyDistance,
                               int gunHeat, int action) {
        double[] oneHot = new double[20];
        int ofs = 0;
        for (int i = 0; i < posXDim; i ++) {
            oneHot[i + ofs] = i == posX ? 1 : -1;
        }
        ofs += posXDim;

        for (int i = 0; i < posYDim; i ++) {
            oneHot[i + ofs] = i == posY ? 1 : -1;
        }
        ofs += posYDim;

        for (int i = 0; i < energyDim; i ++) {
            oneHot[i + ofs] = i == energy ? 1 : -1;
        }
        ofs += energyDim;

        for (int i = 0; i < enemyDistanceDim; i ++) {
            oneHot[i + ofs] = i == enemyDistance ? 1 : -1;
        }
        ofs += enemyDistanceDim;

        for (int i = 0; i < gunHeatDim; i ++) {
            oneHot[i + ofs] = i == gunHeat ? 1 : -1;
        }
        ofs += gunHeatDim;

        for (int i = 0; i < actionDim; i ++) {
            oneHot[i + ofs] = i == action ? 1 : -1;
        }

        return oneHot;
    }
}
