package cpen502.robots;

import cpen502.LUT.RoboCodeLUT;
import robocode.*;

import java.io.*;
import java.util.*;

/**
 * This LUT class used for RL Q-value training. The output of this trained LUT can be
 * further used for neural net training.
 *
 * State Space:
 *      posX : 4 states from 0 to BattleFieldWidth
 *      posY : 4 states from 0 to BattleFieldHeight
 *      energy : 2 states of low, high, separated by a threshold
 *      gunHeat : 2 states of heated, cooled
 *      enemyDistance : 3 states of close, median, far, separated by two thresholds
 *
 *      Total state space = 4x4x2x2x3 = 192
 *
 * Action Space
 *      fire
 *      Forward
 *      Backward
 *      ForwardAvoid
 *      BackwardAvoid
 *
 *      Total action space = 5
 *
 * Total State-Action space = 960
 *
 * Input vector for Neural Network:
 *      Here's an example of one input vector
 *            posX              posY     energy   gunHeat   EnemyDistance
 *      | -1, -1, 1, -1 | -1, 1, -1, 1 | -1, 1  | -1, 1  |      1, -1, -1     |
 *      We use bipolar one hot code encoding for the input
 *      The output Q-value is also scaled to [-1, 1]
 */

public class QLearningRobot extends AdvancedRobot {

    public enum Action {
        Fire(0),
        Forward(1),
        Backward(2),
        ForwardAvoid(3),
        BackwardAvoid(4);

        private final int value;
        private Action(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    public enum StateCategory {
        PosX(0), PosY(1),
        Energy(2),
        EnemyDistance(3),
        GunHeated(4);

        private final int value;
        private StateCategory(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    public final static Map<StateCategory, Integer> stateDim =
            new HashMap<StateCategory, Integer>() {{
               put(StateCategory.PosX, 4); put(StateCategory.PosY, 4);
               put(StateCategory.Energy, 2);
               put(StateCategory.EnemyDistance, 3);
               put(StateCategory.GunHeated, 2);
            }};


    public final static int actionNum = 5;
    public final static int stateNum = 5;

    final static double energyThreshold = 50;
    final static double distanceThreshold1 = 0.2; // 0.2 * max distance
    final static double distanceThreshold2 = 0.4;  // 0.4 * max distance

    final double speed = 40; // pixel/turn

    final boolean interReward = true;
    final double bulletHitBulletReward = -1;
    final double bulletHitRobotReward = 5;
    final double bulletHitWallReward = -1;
    final double hitByBulletReward = -5;
    final double hitByRobotReward = -5;
    final double hitByWallReward = -10;
    final double deathReward = -50;
    final double winReward = 50;
    final double scanRobotReward = 0;

    static double learningRate = 0.001;
    static double featureFactor = 0.8;
    static double explorationRate = 0.5;
    static double initialERate = explorationRate;
    static boolean useOffPolicy = true;
    static boolean useNN = true;
    static int lastNSize = 4;
    static boolean loadNNFile = false;
    static boolean saveNNFile = false;

    final static RoboCodeLUT lut =
            new RoboCodeLUT(learningRate, featureFactor, explorationRate, useOffPolicy,
                    useNN, lastNSize);
    int[] prevState = null;
    Action prevAction = null;

    static int roundCount = 0;
    static List<Double> winRates = new ArrayList<>();
    static List<Double> eRates = new ArrayList<>();
    static List<Integer> rounds = new ArrayList<>();
    static List<Double> rewards = new ArrayList<>();
    static double sumReward = 0;
    static int sumRewardCounter = 0;
    static int winCount = 0;

    double reward = 0;

    static boolean saveFile = false;
    static boolean loadFile = false;

    public void run() {
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForRobotTurn(true);
        execute();

        // read data from file
        if (roundCount == 0 && loadFile) {
            lut.load(getDataFile("LUT.txt"));
        }

        if (roundCount == 0 && useNN && loadNNFile) {
            lut.neuralnet.load(getDataFile("NN.txt"));
        }

        while (true) {
            // reduce exploration rate by time
            if (getRoundNum() - roundCount == 100) {
                roundCount = getRoundNum();
                explorationRate -= 0.01;
                explorationRate = explorationRate < 0 ? 0 : explorationRate;
                lut.setE(explorationRate);
                winRates.add((double)winCount/100);
                rounds.add(roundCount);
                eRates.add(explorationRate);
                winCount = 0;
            }
            turnRadarRight(360);
        }
    }

    public void onBattleEnded(BattleEndedEvent event) {
        if (saveFile) lut.save(getDataFile("LUT.txt"));
        if (saveNNFile) lut.neuralnet.save(getDataFile("NN.txt"));

        String policyString = useOffPolicy ? "offPolicy" : "onPolicy";
        String NNString = useNN ? "NN-" : "";
        PrintStream w = null;
        try {
            out.println("Try writing result to: " + getDataDirectory());

            w = new PrintStream(new RobocodeFileOutputStream(
                    getDataFile("./" + NNString + initialERate + "-" +
                            featureFactor + "-" + lastNSize + "-" + policyString +".txt")));
            for (int i = 0; i < rounds.size(); i ++) {
                w.println(rounds.get(i) + "," + eRates.get(i) + "," + winRates.get(i));
            }

            w = new PrintStream(new RobocodeFileOutputStream(
                    getDataFile("./" + NNString + initialERate + "-" +
                            featureFactor + "-" + lastNSize + "-" + policyString + "-rewards.txt")
            ));
            for (int i = 0; i < rewards.size(); i++) {
                w.println(rewards.get(i));
            }

            if (w.checkError()) {
                out.println("I could not write the count!");
            }

        } catch (IOException e) {
            out.println("IOException trying to write: ");
            e.printStackTrace(out);
        } finally {
            if (w != null) {
                w.close();
            }
        }
    }

    public void onScannedRobot(ScannedRobotEvent event) {
//        aimEnemy(event);
        if (interReward) reward += scanRobotReward;
        int[] state = evaluateState(event);
        Action action = useNN ? lut.updateValueNN(state, prevState, prevAction, reward) :
                lut.updateValue(state, prevState, prevAction, reward);
        performAction(action, event);

        if (sumRewardCounter < 200) {
            sumReward += reward;
            sumRewardCounter ++;
        } else {
            rewards.add(sumReward);
            sumRewardCounter = 0;
            sumReward = 0;
        }

        reward = 0.0;
        prevState = state;
        prevAction = action;
    }

    public void onBulletHit(BulletHitEvent event) {
//        if (interReward) reward += bulletHitRobotReward * event.getBullet().getPower();
        if (interReward) reward += bulletHitRobotReward;
    }

    public void onBulletHitBullet(BulletHitBulletEvent event) {
//        if (interReward) reward += bulletHitBulletReward * event.getBullet().getPower();
        if (interReward) reward += bulletHitBulletReward;
    }

    public void onBulletMissed(BulletMissedEvent event) {
//        if (interReward) reward += bulletHitWallReward * event.getBullet().getPower();
        if (interReward) reward += bulletHitWallReward;
    }

    public void onHitByBullet(HitByBulletEvent event) {
//        if (interReward) reward += hitByBulletReward * event.getBullet().getPower();
        if (interReward) reward += hitByBulletReward;
    }

    public void onHitRobot(HitRobotEvent event) {
        if (interReward) reward += hitByRobotReward;
    }

    public void onHitWall(HitWallEvent event) {
        if (interReward) reward += hitByWallReward;
    }

    public void onRobotDeath(RobotDeathEvent event) {
        reward += winReward;
        winCount ++;
    }

    public void onDeath(DeathEvent event) {
        reward += deathReward;
    }

    public void onWin(WinEvent event) {
        reward += winReward;
    }

    private void performAction(Action action, ScannedRobotEvent event) {
        switch (action) {
            case Fire:
                actionFire(event);
                break;
            case Forward:
                actionForward(event);
                break;
            case Backward:
                actionBackward(event);
                break;
            case ForwardAvoid:
                actionForwardAvoid(event);
                break;
            case BackwardAvoid:
                actionBackwardAvoid(event);
                break;
            default:
                System.out.println("Invalid Action " + action);
        }
    }

    private int[] evaluateState(ScannedRobotEvent event) {
        int[] state = new int[stateNum];
        state[StateCategory.PosX.getValue()] = (int) (stateDim.get(StateCategory.PosX) *
                (getX() / getBattleFieldWidth()));
        state[StateCategory.PosY.getValue()] = (int) (stateDim.get(StateCategory.PosY) *
                (getY() / getBattleFieldHeight()));

        state[StateCategory.Energy.getValue()] = getEnergy() < energyThreshold ? 0 : 1;

        state[StateCategory.EnemyDistance.getValue()] =
                event.getDistance() < distanceThreshold1 ? 0 :
                        event.getDistance() < distanceThreshold2 ? 1 : 2;

        state[StateCategory.GunHeated.getValue()] = getGunHeat() > 0 ? 1 : 0;

        return state;
    }

    private void actionFire(ScannedRobotEvent event) {

        aimEnemy(event);

        if (event.getDistance() > 600) {
            setFire(1.5);
        } else if (event.getDistance() > 400) {
            setFire(2);
        } else {
            setFire(3);
        }
    }

    private void actionForward(ScannedRobotEvent event) {
        setTurnRight(event.getBearing());
        setAhead(speed);
    }

    private void actionBackward(ScannedRobotEvent event) {
        setTurnRight(event.getBearing());
        setBack(speed);
    }

    private void actionForwardAvoid(ScannedRobotEvent event) {
        setTurnRight(event.getBearing() + 90);
        setAhead(speed);
    }

    private void actionBackwardAvoid(ScannedRobotEvent event) {
        setTurnRight(event.getBearing() + 90);
        setBack(speed);
    }

    // aim radar and gun to the enemy
    private void aimEnemy(ScannedRobotEvent event) {
        double targetHeading = getHeading() + event.getBearing();
        if (targetHeading < 0) {
            targetHeading += 360;
        } else if (targetHeading > 360) {
            targetHeading -= 360;
        }

        double gunHeading = getGunHeading();
        double deltaBearing = Math.abs(targetHeading - gunHeading);
        if (deltaBearing > 180) {
            deltaBearing = 360 - 180;
            if (gunHeading > targetHeading) {
                turnGunRight(deltaBearing);
            } else {
                turnGunLeft(deltaBearing);
            }
        } else {
            if (gunHeading > targetHeading) {
                turnGunLeft(deltaBearing);
            } else {
                turnGunRight(deltaBearing);
            }
        }
    }

}
