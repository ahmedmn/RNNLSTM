/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rnnlstm;

/**
 *
 * @author ahmed
 */
public class Vector {

    Sigmoid sigmoid = new Sigmoid();
    
    public Vector() {

    }

    public double vectorVectorMultDot(double[] vect1, double[] vect2) {
        if (vect1.length != vect2.length) {
            throw new IllegalArgumentException("Dimension mismatch, vector lengths are "
                    + vect1.length + " and " + vect2.length + ".");
        }
        double result = 0;
        for (int i = 0; i < vect1.length; i++) {
            result += vect1[i] * vect2[i];
        }
        return result;
    }

    public double[] vectorVectorMultAsterisk(double[] vect1, double[] vect2) {
        if (vect1.length != vect2.length) {
            throw new IllegalArgumentException("Dimension mismatch, vector lengths are "
                    + vect1.length + " and " + vect2.length + ".");
        }
        double[] result = new double[vect1.length];
        for (int i = 0; i < vect1.length; i++) {
            result[i] = vect1[i] * vect2[i];
        }
        return result;
    }

    public double[][] vectorVectorMultDotM(double[] vect1, double[] vect2) {
        double[][] result = new double[vect1.length][];
        for (int i = 0; i < vect1.length; i++) {
            result[i] = new double[vect2.length];
            for (int j = 0; j < vect2.length; j++) {
                result[i][j] = vect1[i] * vect2[j];
            }
        }
        return result;
    }

    public void addVectors(double[] vect1, double[] vect2) {
        if (vect1.length != vect2.length) {
            throw new IllegalArgumentException("Dimension mismatch, vector lengths are "
                    + vect1.length + " and " + vect2.length + ".");
        }
        for (int i = 0; i < vect1.length; i++) {
            vect1[i] += vect2[i];
        }
    }

    public double[] vectSigmoid(double[] vect) {
        double[] result = new double[vect.length];
        for (int i = 0; i < vect.length; i++) {
            result[i] = sigmoid.computeSigmoid(vect[i]);
        }
        return result;
    }

    public double[] vectSigmoidOutputToDerivative(double[] vect) {
        double[] result = new double[vect.length];
        for (int i = 0; i < vect.length; i++) {
            result[i] = sigmoid.sigmoidOutputToDerivative(vect[i]);
        }
        return result;
    }

    public double[] scalarVectMult(double scalar, double[] vect) {
        double[] result = new double[vect.length];
        for (int i = 0; i < vect.length; i++) {
            result[i] = scalar * vect[i];
        }
        return result;
    }

}
