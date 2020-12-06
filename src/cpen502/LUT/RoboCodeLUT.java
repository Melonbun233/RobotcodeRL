package cpen502.LUT;

import cpen502.nerualnetwork.NeuralNetwork;
import cpen502.robots.QLearningRobot;
import cpen502.robots.QLearningRobot.StateCategory;
import cpen502.robots.QLearningRobot.Action;
import cpen502.utils.Functions;
import robocode.RobocodeFileOutputStream;
import robocode.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

public class RoboCodeLUT {
    public final static int lutDepth = QLearningRobot.stateNum;
    public final static int actionDim = QLearningRobot.actionNum;

    public final static Map<QLearningRobot.StateCategory, Integer> stateDim = QLearningRobot.stateDim;

    public final static int posXDim = stateDim.get(StateCategory.PosX);
    public final static int posYDim = stateDim.get(StateCategory.PosY);
    public final static int energyDim = stateDim.get(StateCategory.Energy);
    public final static int enemyDistanceDim = stateDim.get(StateCategory.EnemyDistance);
    public final static int gunHeatDim = stateDim.get(StateCategory.GunHeated);

    private double alpha; // learning rate
    private double gamma; // feature discount factor (0,1)
    private double e;     // exploration rate
    private boolean useOffPolicy;

    private int lastNSize;
    private int lastNCount;
    private int lastNHead;
    private double[][] lastNInputVectors; // prev state and prev action
    private int[][] lastNState; // cur state
    private double[] lastNReward; // reward


    private boolean useNN;
    public NeuralNetwork neuralnet = null;

    private Random rand = new Random();

    double[][][][][][] lut = new double[posXDim][posYDim][energyDim]
            [enemyDistanceDim][gunHeatDim][actionDim];

    int[][][][][][] access = new int[posXDim][posYDim][energyDim]
            [enemyDistanceDim][gunHeatDim][actionDim];

    public RoboCodeLUT(double learningRate, double featureFactor, double explorationRate,
                       boolean useOffPolicy, boolean useNN, int lastNSize) {
        initialize();
        this.alpha = learningRate;
        this.gamma = featureFactor;
        this.e = explorationRate;
        this.useOffPolicy = useOffPolicy;
        this.useNN = useNN;
        this.lastNSize = lastNSize;

        if (useNN) {
            Function<Double, Double> activationFunction = Functions.sigmoidBipolar;
            Function<Double, Double> activationDerivativeFunction = Functions.sigmoidDerivativeBipolar;
            int hiddenNeuronNum = 80;
            double momentums = 0.6;
            double learningRates = 0.01;
            int[] neuronNums = new int[] {20, hiddenNeuronNum, 1};
            try {
                neuralnet = new NeuralNetwork(neuronNums, activationFunction,
                        activationDerivativeFunction, momentums, learningRates);
                neuralnet.initializeWeights();
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
            lastNInputVectors = new double[lastNSize][20];
            lastNState = new int[lastNSize][5];
            lastNReward = new double[lastNSize];
            lastNHead = 0;
            lastNCount = 0;
        }
    }

    public Action updateValueNN(int[] curState, int[] prevState, Action prevAction, double reward) {
        Action optimalAction = null;
        double optimalQ = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < actionDim; i ++) {
            double q = neuralnet.outputFor(SAToOneHot(curState, Action.values()[i]))[0];
            if (q > optimalQ) {
                optimalQ = q;
                optimalAction = Action.values()[i];
            }
        }

        boolean takeExploration = rand.nextDouble() <= e;
        Action randomAction = null;
        double randomQ = 0;
        if (takeExploration) {
            int actionIndex = rand.nextInt(actionDim);
            randomAction = Action.values()[actionIndex];
            randomQ = neuralnet.outputFor(SAToOneHot(curState, randomAction))[0];
        }

        double[] inputVector;
        if (prevState != null) {
            inputVector = SAToOneHot(prevState, prevAction);
            double prevQ = neuralnet.outputFor(inputVector)[0];
            double newQ;
            if (useOffPolicy || !takeExploration) {
                newQ = prevQ + alpha * (reward + gamma * optimalQ - prevQ);
            } else {
                newQ = prevQ + alpha * (reward + gamma * randomQ - prevQ);
            }
            neuralnet.train(inputVector, new double[]{newQ});


            if (lastNSize > 0) {
                trainLastN();
                if (lastNCount == lastNSize) {
                    // replace head with the new values and add 1 to head
                    lastNInputVectors[lastNHead] = inputVector;
                    lastNReward[lastNHead] = reward;
                    lastNState[lastNHead] = curState;
                    lastNHead = (lastNHead + 1) % lastNSize;
                } else {
                    // simply put the value to the tail
                    lastNInputVectors[(lastNHead + lastNCount) % lastNSize] = inputVector;
                    lastNReward[(lastNHead + lastNCount) % lastNSize] = reward;
                    lastNState[(lastNHead + lastNCount) % lastNSize] = curState;
                    lastNCount ++;
                }
            }
        }

        if (takeExploration) {
            return randomAction;
        } else {
            return optimalAction;
        }
    }

    private void trainLastN() {
        // train the last N vectors
        for (int i = 0; i < lastNCount; i ++) {
            double[] inputVector = lastNInputVectors[(lastNHead + i) % lastNSize];
            int[] curState = lastNState[(lastNHead + i) % lastNSize];
            double reward = lastNReward[(lastNHead + i) % lastNSize];

            double optimalQ = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < actionDim; j ++) {
                double q = neuralnet.outputFor(SAToOneHot(curState, Action.values()[j]))[0];
                if (q > optimalQ) {
                    optimalQ = q;
                }
            }

            boolean takeExploration = rand.nextDouble() <= e;
            Action randomAction = null;
            double randomQ = 0;
            if (takeExploration) {
                int actionIndex = rand.nextInt(actionDim);
                randomAction = Action.values()[actionIndex];
                randomQ = neuralnet.outputFor(SAToOneHot(curState, randomAction))[0];
            }

            double prevQ = neuralnet.outputFor(inputVector)[0];
            double newQ;
            if (useOffPolicy || !takeExploration) {
                newQ = prevQ + alpha * (reward + gamma * optimalQ - prevQ);
            } else {
                newQ = prevQ + alpha * (reward + gamma * randomQ - prevQ);
            }
            neuralnet.train(inputVector, new double[]{newQ});
        }
    }




    /**
     * Update the Q value of previous state given the current state and immediate reward
     * Return the action selected for the current state
     * @param curState denotes current state
     * @param prevState denotes previous state
     * @param reward
     */
    public Action updateValue(int[] curState, int[] prevState, Action prevAction, double reward) {

        // get optimal action
        double[] actionSpace = getActionSpace(curState);
        Action optimalAction = getOptimalAction(actionSpace);

        // decide whether to take the exploration move
        boolean takeExploration = rand.nextDouble() <= e;
        Action randomAction = null;
        if (takeExploration) {
            randomAction = getRandomAction(actionSpace);
        }

        // update prev state-action pair based on the policy
        // prevState could be null when it's the first step
        if (prevState != null) {
            double[] prevActionSpace = getActionSpace(prevState);
            if (useOffPolicy || !takeExploration) {
                prevActionSpace[prevAction.getValue()] = prevActionSpace[prevAction.getValue()] + alpha * (reward +
                        gamma * actionSpace[optimalAction.getValue()] - prevActionSpace[prevAction.getValue()]);
            } else {
                prevActionSpace[prevAction.getValue()] = prevActionSpace[prevAction.getValue()] + alpha * (reward +
                        gamma * actionSpace[randomAction.getValue()] - prevActionSpace[prevAction.getValue()]);
            }
        }

        int[] access = getAccess(curState);
        // return the action for this state
        if (takeExploration) {
            access[randomAction.getValue()] ++;
            return randomAction;
        } else {
            access[optimalAction.getValue()] ++;
            return optimalAction;
        }

    }

    public void initialize() {
        // holy
        for (int a = 0; a < posXDim; a ++) {
            for (int b = 0; b < posYDim; b ++) {
//                for (int c = 0; c < headingDim; c ++) {
                    for (int d = 0; d < energyDim; d ++) {
                        for (int e = 0; e < enemyDistanceDim; e ++) {
//                            for (int f = 0; f < enemyBearingDim; f ++) {
                                for (int g = 0; g < gunHeatDim; g ++) {
                                    for (int i = 0; i < actionDim; i ++) {
                                        lut[a][b][d][e][g][i] = rand.nextDouble() * 100;
                                        access[a][b][d][e][g][i] = 0;
                                    }
                                }
//                            }
                        }
                    }
//                }
            }
        }
    }

    private double[] SAToOneHot(int[] state, Action action) {
        double[] oneHot = new double[20];
        int ofs = 0;
        for (int i = 0; i < posXDim; i ++) {
            oneHot[i + ofs] = i == state[0] ? 1 : -1;
        }
        ofs += posXDim;

        for (int i = 0; i < posYDim; i ++) {
            oneHot[i + ofs] = i == state[1] ? 1 : -1;
        }
        ofs += posYDim;

        for (int i = 0; i < energyDim; i ++) {
            oneHot[i + ofs] = i == state[2] ? 1 : -1;
        }
        ofs += energyDim;

        for (int i = 0; i < enemyDistanceDim; i ++) {
            oneHot[i + ofs] = i == state[3] ? 1 : -1;
        }
        ofs += enemyDistanceDim;

        for (int i = 0; i < gunHeatDim; i ++) {
            oneHot[i + ofs] = i == state[4] ? 1 : -1;
        }
        ofs += gunHeatDim;

        for (int i = 0; i < actionDim; i ++) {
            oneHot[i + ofs] = i == action.getValue() ? 1 : -1;
        }

        return oneHot;
    }

    /**
     * Take the current states and return a random move based on the exploration rate
     */
    private Action getRandomAction(double[] actionSpace) {
        if (actionSpace.length != actionDim) {
            return null;
        }

        int actionNum = rand.nextInt(actionDim);
        for (Action action : Action.values()) {
            if (actionNum == action.getValue()) {
                return action;
            }
        }

        return null;
    }

    /**
     * Take the current states and return an action that maximize the Q value
     */
    private Action getOptimalAction(double[] actionSpace) {
        if (actionSpace.length != actionDim) {
            return null;
        }

        int optimalActionNum = 0;
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < actionDim; i ++) {
            if (maxValue < actionSpace[i]) {
                maxValue = actionSpace[i];
                optimalActionNum = i;
            }
        }

        for (Action action : Action.values()) {
            if (optimalActionNum == action.getValue()) {
                return action;
            }
        }

        return null;
    }

    private double[] getActionSpace(int[] state) {
        if (state.length != lutDepth) {
            return null;
        }

        return lut[state[0]][state[1]][state[2]][state[3]][state[4]];
    }

    private int[] getAccess(int[] state) {
        if (state.length != lutDepth) {
            return null;
        }
        return access[state[0]][state[1]][state[2]][state[3]][state[4]];
    }

    public double getAlpha() {
        return alpha;
    }

    public double getGamma() {
        return gamma;
    }

    public double getE() {
        return e;
    }

    public boolean getUseOffPolicy() {
        return useOffPolicy;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }

    public void setE(double e) {
        this.e = e;
    }

    public void setUseOffPolicy(boolean useOffPolicy) {
        this.useOffPolicy = useOffPolicy;
    }

    // Some IO functions

    /**
     * A method to write either a LUT or weights of an neural net to a file.
     * @param argFile of type File.
     */
    public void save(File argFile) {
        PrintStream ps = null;
        try {
            ps = new PrintStream(new RobocodeFileOutputStream(argFile));
            for (int a = 0; a < posXDim; a ++) {
                for (int b = 0; b < posYDim; b ++) {
                    for (int c = 0; c < energyDim; c ++) {
                        for (int d = 0; d < enemyDistanceDim; d ++) {
                            for (int e = 0; e < gunHeatDim; e ++) {
                                for (int f = 0; f < actionDim; f ++) {
                                    ps.println(lut[a][b][c][d][e][f]);
                                }
                            }
                        }
                    }
                }
            }
            ps.close();
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }
        return;
    }
    /**
     * Loads the LUT or neural net weights from file. The load must of course
     * have knowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being
     * made to load data into an LUT or neural net whose structure does not match
     * the data in the file. (e.g. wrong number of hidden neurons).
     * @throws IOException
     */
    public void load(File argFile) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(argFile));
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
        return;
    }
}
