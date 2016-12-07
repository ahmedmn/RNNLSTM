package rnnlstm;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Random;
import static rnnlstm.BinaryAdding.addVectors;
import static rnnlstm.BinaryAdding.computeSigmoid;
import static rnnlstm.BinaryAdding.sigmoidOutputToDerivative;
import static rnnlstm.BinaryAdding.vectSigmoid;
import static rnnlstm.BinaryAdding.vectorMatrixMult;
import static rnnlstm.BinaryAdding.vectorVectorMultDot;

public class RNNLSTM {

    public static int[][] LoadData(List<String> list) {
        String[] arraystr = list.toArray(new String[list.size()]);

        int[][] result = new int[arraystr.length][];

        for (int i = 0; i < arraystr.length; i++) {
            String[] row = arraystr[i].split(",");
            result[i] = new int[row.length];

            for (int j = 0; j < row.length; j++) {
                result[i][j] = Integer.parseInt(row[j]);
            }
        }
        return result;
    }

    public static void main(String[] args) throws IOException {

        DecimalFormat df = new DecimalFormat("###.##");
        List<String> X_trainStr = Files.readAllLines(Paths.get("E:\\Work\\MasarykUni\\NeuralNetworks\\Project\\RNNLSTM\\src\\Data\\X_train.csv"));
        List<String> y_trainStr = Files.readAllLines(Paths.get("E:\\Work\\MasarykUni\\NeuralNetworks\\Project\\RNNLSTM\\src\\Data\\y_train.csv"));

        List<String> X_testStr = Files.readAllLines(Paths.get("E:\\Work\\MasarykUni\\NeuralNetworks\\Project\\RNNLSTM\\src\\Data\\X_test.csv"));
        List<String> y_testStr = Files.readAllLines(Paths.get("E:\\Work\\MasarykUni\\NeuralNetworks\\Project\\RNNLSTM\\src\\Data\\y_test.csv"));

        int[][] X_train = LoadData(X_trainStr);
        int[] y_train = Arrays.asList(y_trainStr.toArray(new String[y_trainStr.size()])).stream().mapToInt(Integer::parseInt).toArray();
        int[][] X_test = LoadData(X_testStr);
        int[] y_test = Arrays.asList(y_testStr.toArray(new String[y_testStr.size()])).stream().mapToInt(Integer::parseInt).toArray();

        System.out.println("Traing Sample");
        System.out.print("[");
        for (int i = 0; i < X_train[0].length; i++) {
            System.out.print(X_train[0][i] + " ");
        }
        System.out.println("]" + "\n" + y_train[0]);

        System.out.println("Testing Sample");
        System.out.print("[");
        for (int i = 0; i < X_test[0].length; i++) {
            System.out.print(X_test[0][i] + " ");
        }
        System.out.println("]" + "\n" + y_test[0]);

//        List<ArrayList> X_train = new ArrayList<ArrayList>(Arrays.asList(X_trainStr.split))
        Sigmoid sigmoid = new Sigmoid();
        Matrix matrix = new Matrix();
        Vector vector = new Vector();

//        int binaryDim = 8;
//        int largestNumber = (int) Math.pow(2, binaryDim);
        double alpha = 0.005;
        int inputDim = 200;
        int hiddenDim = inputDim;

        int dataRownum = X_train.length;
        int datacolnum = X_train[0].length;

        double[] weightsOut = new double[inputDim];
        double[] weightsOutUpdate = new double[inputDim];
        for (int i = 0; i < inputDim; i++) {
            weightsOut[i] = Math.random() * 2 - 1;
            weightsOutUpdate[i] = 0;
        }
        
        double[] weightsIn = new double[inputDim];
        double[] weightsInUpdate = new double[inputDim];
        for (int i = 0; i < inputDim; i++) {
            weightsIn[i] = Math.random() * 2 - 1;
            weightsInUpdate[i] = 0;

//            weightsIn[i] = new double[hiddenDim];
//            weightsInUpdate[i] = new double[hiddenDim];
//            for (int j = 0; j < hiddenDim; j++) {
//                weightsIn[i][j] = Math.random() * 2 - 1;
//                weightsInUpdate[i][j] = 0;
//            }
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

        for (int j = 0; j < 250; j++) {

            double[] d = new double[dataRownum];

            double overallError = 0;

            double[][] layer1Values = new double[dataRownum + 1][];
            layer1Values[0] = new double[hiddenDim];
            for (int k = 0; k < hiddenDim; k++) {
                layer1Values[0][k] = 0;
            }

            double[] layer1, x, temp;
            double layer2, y, layer2Error, layer2Delta;
            for (int position = 0; position < dataRownum; position++) {
                x = new double[datacolnum];
                for (int col = 0; col < datacolnum; col++) {
                    x[col] = X_train[position][col];
                }
//                x[0] = a.get(binaryDim - position - 1) ? 1 : 0;
//                x[1] = b.get(binaryDim - position - 1) ? 1 : 0;

                // y = c.get(binaryDim - position - 1) ? 1 : 0;
                y = y_train[position];

                for (int col = 0; col < datacolnum; col++) {
                    temp = vector.scalarVectMult(x[col], weightsIn);
                    vector.addVectors(temp, matrix.vectorMatrixMult(layer1Values[col], weightsHidden));
                    layer1 = vector.vectSigmoid(temp); //memory vector

                    layer1Values[col + 1] = layer1;
                }

                //compute output
                layer2 = sigmoid.computeSigmoid(vector.vectorVectorMultDot(layer1Values[100], weightsOut));

                layer2Error = y - layer2;
                d[position] = Math.round(layer2);
                layer2Delta = (layer2Error * sigmoid.sigmoidOutputToDerivative(layer2)); //maybe consider Math.abs if it does not work

                addVectors(weightsOutUpdate, vector.scalarVectMult(layer2Delta, layer1Values[datacolnum]));

                
                double[] futureLayer1Delta = vector.scalarVectMult(layer2Delta, weightsOut);

                
                double[] prevLayer1, layer1Delta;
                //backpropagation
                for (int col = datacolnum-1; col > 0; col--) {
//                x = new double[2];
//                x[0] = a.get(binaryDim - position) ? 1 : 0;
//                x[1] = b.get(binaryDim - position) ? 1 : 0;
//                    x = new double[datacolnum]; //our old x is still the same :)
//                    for (int col = 0; col < datacolnum; col++) {
//                        x[col] = X_train[dataRownum - position][col];
//                    }

                    layer1 = layer1Values[col];
                    prevLayer1 = layer1Values[col - 1];


                    layer1Delta = vector.vectorVectorMultAsterisk(futureLayer1Delta,
                                                                  vector.vectSigmoidOutputToDerivative(layer1));

                    matrix.addMatrices(weightsHiddenUpdate, vector.vectorVectorMultDotM(prevLayer1, layer1Delta));
                    vector.addVectors(weightsInUpdate, vector.scalarVectMult(x[col], layer1Delta));

                    futureLayer1Delta = layer1Delta;
                }

                vector.addVectors(weightsOut, vector.scalarVectMult(alpha, weightsOutUpdate));
                matrix.addMatrices(weightsHidden, matrix.scalarMatrixMult(alpha, weightsHiddenUpdate));
                vector.addVectors(weightsIn, vector.scalarVectMult(alpha, weightsInUpdate));

                //TODO do we need to erase these vectors and matrices?
                for (int i = 0; i < weightsOutUpdate.length; i++) {
                    weightsOutUpdate[i] = 0;
                }
                for (int i = 0; i < weightsHiddenUpdate.length; i++) {
                    for (int k = 0; k < weightsHiddenUpdate[0].length; k++) {
                        weightsHiddenUpdate[i][k] = 0;
                    }
                }
                for (int i = 0; i < weightsInUpdate.length; i++) {
                    weightsInUpdate[i] = 0;
                }

                if (j % 1000 == 0) {
                    System.out.println("Error: " + overallError);
//                System.out.println("Pred: " + toString(d));
//                System.out.println("True: " + toString(c, binaryDim));
//                int out = 0;
//                for (int i = 0; i < binaryDim; i++) {
//                    out += d[binaryDim - i - 1] * ((int) Math.pow(2, i));
//                }
//                System.out.print(aInt + " + " + bInt + " = " + out);
//                if (out != cInt) {
//                    System.out.println("\t\t\t\t\tWRONG");
//                }
//                System.out.println("\n------------");

                    System.out.print("Pred: [");
                    for (int i = 0; i < 50; i++) {
                        System.out.print((int) d[i] + " ");
                    }
                    System.out.println("]");

                    System.out.print("Actu: [");
                    for (int i = 0; i < 50; i++) {
                        System.out.print(y_train[i] + " ");
                    }
                    System.out.println("]");

                    System.out.println("Accuracy: " + df.format(getPercentage(y_train, d)) + " %");
                }

            }

        }
    }
    

    public static double getPercentage(int[] arrayA, double[] arrayB) {
        double percentage = 0;
        for (int i = 0; i < arrayA.length; i++) {
            if (arrayA[i] == (int) arrayB[i]) {
                /*note the different indices*/
                percentage++;
                /*count how many times you have matching values*/
 /* NOTE: This only works if you don't have repeating values in arrayA*/
            }

        }

        return (percentage / arrayA.length) * 100;
        /*return the amount of times over the length times 100*/
    }

}
