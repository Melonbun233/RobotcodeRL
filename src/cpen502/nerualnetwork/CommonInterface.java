package cpen502.nerualnetwork;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {
    /**
     * @param X The input vector. An array of doubles.
     * @return The value returned by th LUT or NN for this input vector */
    public double outputFor(double [] X);

    /**
     * This method will tell the NN or the LUT the output
     * value that should be mapped to the given input
     * the desired correct output value for an input.
     * @param X The input vector
     * @param argValue The new value to learn
     * @return The error in the output for that input
     */
    public double train(double [] X, double argValue);

    /**
     * A method to write either a LUT or weights of an neural net to a file.
     * @param argFile of type File.
     */
    public void save(File argFile);

    /**
     * Loads the LUT or neural net weights from file. The load must of course
     * have knowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being
     * made to load data into an LUT or neural net whose structure does not match
     * the data in the file. (e.g. wrong number of hidden neurons).
     * @throws IOException
     */
    public void load(String argFileName) throws IOException;
}
