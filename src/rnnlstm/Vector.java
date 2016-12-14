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

    public void vectorVectorMultAsteriskNoOut(double[] result, double[] vect1, double[] vect2) {
        if (vect1.length != vect2.length) {
            throw new IllegalArgumentException("Dimension mismatch, vector lengths are "
                    + vect1.length + " and " + vect2.length + ".");
        }
        for (int i = 0; i < vect1.length; i++) {
            result[i] = vect1[i] * vect2[i];
        }
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

    public void vectorVectorMultDotMNoOutAdd(double[][] result, double[] vect1, double[] vect2) {
        for (int i = 0; i < vect1.length; i++) {
            for (int j = 0; j < vect2.length; j++) {
                result[i][j] += vect1[i] * vect2[j];
            }
        }
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

    public void vectSigmoidNoOut(double[] vect) {
        for (int i = 0; i < vect.length; i++) {
            vect[i] = sigmoid.computeSigmoid(vect[i]);
        }
    }
    
    public double[] vectSigmoidOutputToDerivative(double[] vect) {
        double[] result = new double[vect.length];
        for (int i = 0; i < vect.length; i++) {
            result[i] = sigmoid.sigmoidOutputToDerivative(vect[i]);
        }
        return result;
    }

    public void vectSigmoidOutputToDerivativeNoOut(double[] result, double[] vect) {
        for (int i = 0; i < vect.length; i++) {
            result[i] = sigmoid.sigmoidOutputToDerivative(vect[i]);
        }
    }
    
    public void vectTangentHNoOut(double[] vect) {
        for (int i = 0; i < vect.length; i++) {
            vect[i] = sigmoid.computeTangentH(vect[i]);
        }
    }
 
    public void vectTangentHToDerivativeNoOut(double[] vect) {
        for (int i = 0; i < vect.length; i++) {
            vect[i] = sigmoid.computeTangentHToDerivative(vect[i]);
        }
    }
    
    public double[] scalarVectMult(double scalar, double[] vect) {
        double[] result = new double[vect.length];
        for (int i = 0; i < vect.length; i++) {
            result[i] = scalar * vect[i];
        }
        return result;
    }

    public void scalarVectMultNoOut(double[] result, double scalar, double[] vect) {
        for (int i = 0; i < vect.length; i++) {
            result[i] = scalar * vect[i];
        }
    }
    
    public void copy(double[] out, double[] in) {
        //TODO add exception?
        for(int i = 0; i<in.length;i++) {
            out[i] = in[i];
        }
    }
}
