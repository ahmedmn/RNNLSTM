/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rnnlstm;

import java.util.BitSet;
import java.util.Random;

/**
 *
 * @author ahmed
 */
public class LSTM {

    Sigmoid sigmoid = new Sigmoid();
    Matrix matrix = new Matrix();
    Vector vector = new Vector();

    public LSTM() {

    }

    private String toString(double[] d) {
        String output = "[";
        for (int i = 0; i < d.length - 1; i++) {
            output += d[i] + ", ";
        }
        output += d[d.length - 1] + "]";
        return output;
    }

    private String toString(BitSet d, int numDigits) {
        String output = "[";
        for (int i = 0; i < numDigits - 1; i++) {
            output += (d.get(i) ? 1.0 : 0.0) + ", ";
        }
        output += (d.get(numDigits - 1) ? 1.0 : 0.0) + "]";
        return output;
    }

    public BitSet getBitSet(int value, int dim) {
        BitSet bits = new BitSet();
        int index = 0;
        while (value != 0L) {
            if (value % 2L != 0) {
                bits.set(dim - index - 1);
            }
            ++index;
            value = value >>> 1;
        }
        return bits;
    }

    public void run(int inputDim, int hiddenDim, double alpha) {

        int binaryDim = 8;
        int largestNumber = (int) Math.pow(2, binaryDim);

        double[][] weightsIn = new double[inputDim][];
        double[][] weightsInUpdate = new double[inputDim][];
        for (int i = 0; i < inputDim; i++) {
            weightsIn[i] = new double[hiddenDim];
            weightsInUpdate[i] = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++) {
                weightsIn[i][j] = Math.random() * 2 - 1;
                weightsInUpdate[i][j] = 0;
            }
        }

        double[] weightsOut = new double[hiddenDim];
        double[] weightsOutUpdate = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsOut[i] = Math.random() * 2 - 1;
            weightsOutUpdate[i] = 0;
        }

        double[][] weightsHidden = new double[hiddenDim][];
        double[][] weightsHiddenUpdate = new double[hiddenDim][];
        for (int i = 0; i < hiddenDim; i++) {
            weightsHidden[i] = new double[hiddenDim];
            weightsHiddenUpdate[i] = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++) {
                weightsHidden[i][j] = Math.random() * 2 - 1;
                weightsHiddenUpdate[i][j] = 0;
            }
        }

        Random rand = new Random();

        for (int j = 0; j < 10000; j++) {

            int aInt = rand.nextInt(largestNumber / 2);
            BitSet a = getBitSet(aInt, binaryDim);

            int bInt = rand.nextInt(largestNumber / 2);

            BitSet b = getBitSet(bInt, binaryDim);

            int cInt = aInt + bInt;
            BitSet c = getBitSet(cInt, binaryDim);

            double[] d = new double[binaryDim];

            double overallError = 0;

            double[] layer2Deltas = new double[binaryDim + 1];
            double[][] layer1Values = new double[binaryDim + 1][];
            layer1Values[0] = new double[hiddenDim];
            for (int k = 0; k < hiddenDim; k++) {
                layer1Values[0][k] = 0;
            }

            double[] layer1, x, temp;
            double layer2, y, layer2Error;
            for (int position = 0; position < binaryDim; position++) {

                x = new double[2];
                x[0] = a.get(binaryDim - position - 1) ? 1 : 0;
                x[1] = b.get(binaryDim - position - 1) ? 1 : 0;

                y = c.get(binaryDim - position - 1) ? 1 : 0;

                temp = matrix.vectorMatrixMult(x, weightsIn);
                vector.addVectors(temp, matrix.vectorMatrixMult(layer1Values[position], weightsHidden));
                layer1 = vector.vectSigmoid(temp);

                layer2 = sigmoid.computeSigmoid(vector.vectorVectorMultDot(layer1, weightsOut));

                layer2Error = y - layer2;
                layer2Deltas[position + 1] = layer2Error * sigmoid.sigmoidOutputToDerivative(layer2);
                overallError += Math.abs(layer2Error);

                d[binaryDim - position - 1] = Math.round(layer2);

                layer1Values[position + 1] = layer1;
            }

            double[] futureLayer1Delta = new double[hiddenDim];
            for (int i = 0; i < futureLayer1Delta.length; i++) {
                futureLayer1Delta[i] = 0;
            }

            double[] prevLayer1, layer1Delta;
            double layer2Delta;
            for (int position = binaryDim; position > 0; position--) {
                x = new double[2];
                x[0] = a.get(binaryDim - position) ? 1 : 0;
                x[1] = b.get(binaryDim - position) ? 1 : 0;
                layer1 = layer1Values[position];
                prevLayer1 = layer1Values[position - 1];

                layer2Delta = layer2Deltas[position];

                temp = matrix.vectorMatrixMult(futureLayer1Delta, matrix.transpose(weightsHidden));
                vector.addVectors(temp, vector.scalarVectMult(layer2Delta, weightsOut));
                layer1Delta = vector.vectorVectorMultAsterisk(temp, vector.vectSigmoidOutputToDerivative(layer1));

                vector.addVectors(weightsOutUpdate, vector.scalarVectMult(layer2Delta, layer1));
                matrix.addMatrices(weightsHiddenUpdate, vector.vectorVectorMultDotM(prevLayer1, layer1Delta));
                matrix.addMatrices(weightsInUpdate, vector.vectorVectorMultDotM(x, layer1Delta));

                futureLayer1Delta = layer1Delta;
            }

            vector.addVectors(weightsOut, vector.scalarVectMult(alpha, weightsOutUpdate));
            matrix.addMatrices(weightsHidden, matrix.scalarMatrixMult(alpha, weightsHiddenUpdate));
            matrix.addMatrices(weightsIn, matrix.scalarMatrixMult(alpha, weightsInUpdate));

            for (int i = 0; i < weightsOutUpdate.length; i++) {
                weightsOutUpdate[i] = 0;
            }
            for (int i = 0; i < weightsHiddenUpdate.length; i++) {
                for (int k = 0; k < weightsHiddenUpdate[0].length; k++) {
                    weightsHiddenUpdate[i][k] = 0;
                }
            }
            for (int i = 0; i < weightsInUpdate.length; i++) {
                for (int k = 0; k < weightsInUpdate[0].length; k++) {
                    weightsInUpdate[i][k] = 0;
                }
            }

            if (j % 1000 == 0) {
                System.out.println("Error: " + overallError);
                System.out.println("Pred: " + toString(d));
                System.out.println("True: " + toString(c, binaryDim));
                int out = 0;
                for (int i = 0; i < binaryDim; i++) {
                    out += d[binaryDim - i - 1] * ((int) Math.pow(2, i));
                }
                System.out.print(aInt + " + " + bInt + " = " + out);
                if (out != cInt) {
                    System.out.println("\t\t\t\t\tWRONG");
                }
                System.out.println("\n------------");
            }

        }

    }

}
