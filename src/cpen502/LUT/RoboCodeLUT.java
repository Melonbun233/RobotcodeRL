package cpen502.LUT;

import cpen502.robots.QLearningRobot;
import cpen502.robots.QLearningRobot.StateCategory;
import cpen502.robots.QLearningRobot.Action;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

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

    private Random rand = new Random();

    double[][][][][][] lut = new double[posXDim][posYDim][energyDim]
            [enemyDistanceDim][gunHeatDim][actionDim];

    int[][][][][][] access = new int[posXDim][posYDim][energyDim]
            [enemyDistanceDim][gunHeatDim][actionDim];

    public RoboCodeLUT(double learningRate, double featureFactor, double explorationRate, boolean useOffPolicy) {
        initialize();
        this.alpha = learningRate;
        this.gamma = featureFactor;
        this.e = explorationRate;
        this.useOffPolicy = useOffPolicy;
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
    public void load(String argFileName) throws IOException {
        return;
    }
}
