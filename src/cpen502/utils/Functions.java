package cpen502.utils;

import java.util.function.Function;

public class Functions {
    /**
     * sigmoid function bounded by (0,1)
     */
    public static final Function<Double, Double> sigmoidBinary = (x -> 1.0/(1.0 + Math.pow(Math.E, -x)));

    /**
     * Derivative of sigmoid function bounded by (0,1)
     * input is f(x)
     */
    public static final Function<Double, Double> sigmoidDerivativeBinary = (x -> x * (1.0 - x));

    /**
     * sigmoid function bounded by (-1,1)
     */
    public static final Function<Double, Double> sigmoidBipolar = (x -> (2.0/(1.0 + Math.pow(Math.E, -x))) - 1.0);

    /**
     * derivative of sigmoid function bounded by (-1,1)
     * input is f(x)
     */
    public static final Function<Double, Double> sigmoidDerivativeBipolar = (x -> 0.5 * (1.0 - x) * (1.0 + x));
}
