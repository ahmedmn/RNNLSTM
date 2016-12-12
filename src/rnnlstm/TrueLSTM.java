/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rnnlstm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import static rnnlstm.BinaryAdding.addVectors;

/**
 * Implementation of the most used type of LSTM.
 *
 * @author xkorenc
 */
public class TrueLSTM {

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

        List<String> X_trainStr = Files.readAllLines(Paths.get("/home/xkorenc/Desktop/neuralNetworks/RNNLSTM.git/trunk/src/Data/X_train.csv"));
        List<String> y_trainStr = Files.readAllLines(Paths.get("/home/xkorenc/Desktop/neuralNetworks/RNNLSTM.git/trunk/src/Data/y_train.csv"));

        List<String> X_testStr = Files.readAllLines(Paths.get("/home/xkorenc/Desktop/neuralNetworks/RNNLSTM.git/trunk/src/Data/X_test.csv"));
        List<String> y_testStr = Files.readAllLines(Paths.get("/home/xkorenc/Desktop/neuralNetworks/RNNLSTM.git/trunk/src/Data/y_test.csv"));

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

        double alpha = 0.005;
        int hiddenDim = 200;
        int numOfReviews = 2;
        int numOfIterations = 1000000;
        int numHiddenIters = 1;
        
        int dataRowNum = X_train.length;
        int dataColNum = X_train[0].length;
        double[] d = new double[dataRowNum];        
        double[] outputs = new double[dataRowNum];
        
        //weights for input and corresponding gates
        double[] weightsIn = new double[hiddenDim];
        double[] weightsInUpdate = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsIn[i] = Math.random() * 2 - 1;
            weightsInUpdate[i] = 0;
        }

        double[] weightsForget = new double[hiddenDim];
        double[] weightsForgetUpdate = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsForget[i] = Math.random() * 2 - 1;
            weightsForgetUpdate[i] = 0;
        }

        double[] weightsMemory = new double[hiddenDim];
        double[] weightsMemoryUpdate = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsMemory[i] = Math.random() * 2 - 1;
            weightsMemoryUpdate[i] = 0;
        }

        double[] weightsOut = new double[hiddenDim];
        double[] weightsOutUpdate = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsOut[i] = Math.random() * 2 - 1;
            weightsOutUpdate[i] = 0;
        }

        // weitghts for output of previous stage and corresponding gates
        double[] temp;
        double[] temp2;

        double[][] weightsIn2 = new double[hiddenDim][];
        double[][] weightsInUpdate2 = new double[hiddenDim][];
        for (int i = 0; i < hiddenDim; i++) {
            temp = new double[hiddenDim];
            temp2 = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++) {
                temp[j] = Math.random() * 2 - 1;
                temp2[j] = 0;
            }
            weightsIn2[i] = temp;
            weightsInUpdate2[i] = temp2;

        }

        double[][] weightsForget2 = new double[hiddenDim][];
        double[][] weightsForgetUpdate2 = new double[hiddenDim][];
        for (int i = 0; i < hiddenDim; i++) {
            temp = new double[hiddenDim];
            temp2 = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++) {
                temp[j] = Math.random() * 2 - 1;
                temp2[j] = 0;
            }
            weightsForget2[i] = temp;
            weightsForgetUpdate2[i] = temp2;

        }

        double[][] weightsMemory2 = new double[hiddenDim][];
        double[][] weightsMemoryUpdate2 = new double[hiddenDim][];
        for (int i = 0; i < hiddenDim; i++) {
            temp = new double[hiddenDim];
            temp2 = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++) {
                temp[j] = Math.random() * 2 - 1;
                temp2[j] = 0;
            }
            weightsMemory2[i] = temp;
            weightsMemoryUpdate2[i] = temp2;

        }

        double[][] weightsOut2 = new double[hiddenDim][];
        double[][] weightsOutUpdate2 = new double[hiddenDim][];
        for (int i = 0; i < hiddenDim; i++) {
            temp = new double[hiddenDim];
            temp2 = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++) {
                temp[j] = Math.random() * 2 - 1;
                temp2[j] = 0;
            }
            weightsOut2[i] = temp;
            weightsOutUpdate2[i] = temp2;

        }

        //final output weights
        double[] weightsFinal = new double[hiddenDim];
        double[] weightsFinalUpdate = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsFinal[i] = Math.random() * 2 - 1;
            weightsFinalUpdate[i] = 0;
        }

        Random rand = new Random();

        int counter =0;
        for (int j = 0; j < numOfIterations; j++) {



            double[] layer1, x;
            double[][] forgetGate, inputGate, outputGate, memoryInput, memory, output, almostOutput;
            forgetGate = new double[dataColNum + 1][]; //f^t
            inputGate = new double[dataColNum + 1][]; //i^t
            outputGate = new double[dataColNum + 1][]; //o^t
            memoryInput = new double[dataColNum + 1][]; //a^t
            memory = new double[dataColNum + 1][]; //c^t
            output = new double[dataColNum + 1][]; //h^t
            almostOutput = new double[dataColNum + 1][];

            memory[0] = new double[hiddenDim];
            output[0] = new double[hiddenDim];
            for (int k = 0; k < hiddenDim; k++) {
                memory[0][k] = 0;
                output[0][k] = 0;
            }

            for (int k = 0; k < dataColNum+1; k++) {
                almostOutput[k] = new double[hiddenDim];
            }

            double finalOutput, y, finalOutputError, finalOutputDelta;
            

            for (int reviewRow = 0; reviewRow < numOfReviews; reviewRow++) { //TODO erase
//            for (int reviewRow = 0; reviewRow < dataRowNum; reviewRow++) {
                counter++;

                x = new double[dataColNum + 1];
                for (int wordCol = 0; wordCol < dataColNum; wordCol++) {
                    x[wordCol+1] = X_train[reviewRow][wordCol];
                }

                y = y_train[reviewRow];

                //forward pass
                for (int wordCol = 1; wordCol <= dataColNum; wordCol++) {
                    temp = vector.scalarVectMult(x[wordCol], weightsForget);
                    vector.addVectors(temp, matrix.vectorMatrixMult(output[wordCol-1], weightsForget2));
                    vector.vectSigmoidNoOut(temp);
                    forgetGate[wordCol] = temp;
                    memory[wordCol] = vector.vectorVectorMultAsterisk(memory[wordCol-1], temp);

                    temp = vector.scalarVectMult(x[wordCol], weightsIn);
                    vector.addVectors(temp, matrix.vectorMatrixMult(output[wordCol-1], weightsIn2));
                    vector.vectSigmoidNoOut(temp);
                    inputGate[wordCol] = temp; //TODO we won't need to create new arrays for temp/ we can reuse the old ones and keep them in inputGate[wordCol]

                    temp = vector.scalarVectMult(x[wordCol], weightsMemory);
                    vector.addVectors(temp, matrix.vectorMatrixMult(output[wordCol-1], weightsMemory2));
                    vector.vectTangentHNoOut(temp);
                    memoryInput[wordCol] = temp;

                    vector.addVectors(memory[wordCol],
                            vector.vectorVectorMultAsterisk(inputGate[wordCol],
                                    memoryInput[wordCol]));

                    vector.copy(almostOutput[wordCol], memory[wordCol]);
                    vector.vectTangentHNoOut(almostOutput[wordCol]);

                    temp = vector.scalarVectMult(x[wordCol], weightsOut);
                    vector.addVectors(temp, matrix.vectorMatrixMult(output[wordCol-1], weightsOut2));
                    vector.vectSigmoidNoOut(temp);
                    outputGate[wordCol] = temp;

                    output[wordCol] = vector.vectorVectorMultAsterisk(outputGate[wordCol], almostOutput[wordCol]);
                    
//                    double count=0;
//                    for(double res:forgetGate[wordCol]){
//                        count += res;
//                    }
//                    System.out.print(count + " ");
                }
//                System.out.println("");
                
                //compute output
                finalOutput = sigmoid.computeSigmoid(vector.vectorVectorMultDot(output[dataColNum], weightsFinal));

                finalOutputError = Math.pow((y - finalOutput)*10,2);
                d[reviewRow] = Math.round(finalOutput);
                outputs[reviewRow] = finalOutput;
                finalOutputDelta = (finalOutputError * sigmoid.sigmoidOutputToDerivative(finalOutput)); //maybe consider Math.abs if it does not work

                addVectors(weightsFinalUpdate, vector.scalarVectMult(finalOutputDelta, weightsFinal)); //computing dh

                double[] futureOutputDelta = weightsFinalUpdate;
                double[] currentOutputDelta = new double[hiddenDim];
                double[] futureMemoryDelta = new double[hiddenDim];
                double[] currentMemoryDelta = new double[hiddenDim];

                for (int i = 0; i < hiddenDim; i++) {
                    futureMemoryDelta[i] = 0;
                }

                double[] prevLayer1, layer1Delta;

                //backpropagation
                for (int wordCol = dataColNum; wordCol > 0; wordCol--) {

                    temp = vector.vectorVectorMultAsterisk(futureOutputDelta, almostOutput[wordCol]); //do^t
                    temp = vector.vectorVectorMultAsterisk(temp, vector.vectSigmoidOutputToDerivative(outputGate[wordCol]));//d(o^)^t
                    vector.addVectors(weightsOutUpdate, vector.scalarVectMult(x[wordCol], temp)); //TODO check whether it is correct
                    matrix.addMatrices(weightsOutUpdate2, vector.vectorVectorMultDotM(output[wordCol], temp));
                    //computing dh^(t-1)
                    currentOutputDelta = matrix.vectorMatrixMult(temp, matrix.transpose(weightsOut2));

                    temp = vector.vectorVectorMultAsterisk(futureOutputDelta, outputGate[wordCol]); // dh^t . o^t
                    vector.vectTangentHToDerivativeNoOut(almostOutput[wordCol]);
                    temp = vector.vectorVectorMultAsterisk(temp, almostOutput[wordCol]);
                    vector.addVectors(futureMemoryDelta, temp); // dc^t

                    currentMemoryDelta = vector.vectorVectorMultAsterisk(futureMemoryDelta, forgetGate[wordCol]); //dc^(t-1)

                    //forget gate
                    temp = vector.vectorVectorMultAsterisk(futureMemoryDelta, memory[wordCol - 1]); //df^t
                    temp = vector.vectorVectorMultAsterisk(temp, vector.vectSigmoidOutputToDerivative(forgetGate[wordCol]));
                    vector.addVectors(weightsForgetUpdate, vector.scalarVectMult(x[wordCol], temp)); //TODO check whether it is correct
                    matrix.addMatrices(weightsForgetUpdate2, vector.vectorVectorMultDotM(output[wordCol], temp));
                    //computing dh^(t-1)
                    vector.addVectors(currentOutputDelta, matrix.vectorMatrixMult(temp, matrix.transpose(weightsForget2)));

                    //input gate
                    temp = vector.vectorVectorMultAsterisk(futureMemoryDelta, memoryInput[wordCol]); //di^t
                    temp = vector.vectorVectorMultAsterisk(temp, vector.vectSigmoidOutputToDerivative(inputGate[wordCol]));
                    vector.addVectors(weightsInUpdate, vector.scalarVectMult(x[wordCol], temp)); //TODO check whether it is correct
                    matrix.addMatrices(weightsInUpdate2, vector.vectorVectorMultDotM(output[wordCol], temp));
                    //computing dh^(t-1)
                    vector.addVectors(currentOutputDelta, matrix.vectorMatrixMult(temp, matrix.transpose(weightsIn2)));

                    //memory input
                    temp = vector.vectorVectorMultAsterisk(futureMemoryDelta, inputGate[wordCol]); //da^t
                    vector.vectTangentHToDerivativeNoOut(memoryInput[wordCol]);
                    temp = vector.vectorVectorMultAsterisk(temp, memoryInput[wordCol]); //d(a^)^t
                    vector.addVectors(weightsMemoryUpdate, vector.scalarVectMult(x[wordCol], temp)); //TODO check whether it is correct
                    matrix.addMatrices(weightsMemoryUpdate2, vector.vectorVectorMultDotM(output[wordCol], temp));
                    //computing dh^(t-1)
                    vector.addVectors(currentOutputDelta, matrix.vectorMatrixMult(temp, matrix.transpose(weightsMemory2)));

                    // preparing new values of memory and output
                    futureMemoryDelta = currentMemoryDelta;
                    futureOutputDelta = currentOutputDelta;

//                    if (wordCol ==50 && reviewRow % 100 == 0) {
//                        double gradient = 0;
//                        for (int i = 0; i < layer1Delta.length; i++) {
//                            gradient += layer1Delta[i];
//                        }
//                        System.out.println("Gradient at 50-th layer: " + gradient);
//                    }

//                    double count=0;
//                    for(double res:futureMemoryDelta){
//                        count += res;
//                    }
//                    System.out.print(count + " ");
                }
//                System.out.println("");
                
                vector.addVectors(weightsIn, vector.scalarVectMult(alpha, weightsInUpdate));
                vector.addVectors(weightsForget, vector.scalarVectMult(alpha, weightsForgetUpdate));
                vector.addVectors(weightsMemory, vector.scalarVectMult(alpha, weightsMemoryUpdate));
                vector.addVectors(weightsOut, vector.scalarVectMult(alpha, weightsOutUpdate));

                matrix.addMatrices(weightsIn2, matrix.scalarMatrixMult(alpha, weightsInUpdate2));
                matrix.addMatrices(weightsForget2, matrix.scalarMatrixMult(alpha, weightsForgetUpdate2));
                matrix.addMatrices(weightsMemory2, matrix.scalarMatrixMult(alpha, weightsMemoryUpdate2));
                matrix.addMatrices(weightsOut2, matrix.scalarMatrixMult(alpha, weightsOutUpdate2));

                for (int i = 0; i < weightsInUpdate.length; i++) {
                    weightsInUpdate[i] = 0;
                    weightsForgetUpdate[i] = 0;
                    weightsMemoryUpdate[i] = 0;
                    weightsOutUpdate[i] = 0;
                }

                for (int i = 0; i < weightsInUpdate2.length; i++) {
                    for (int k = 0; k < weightsInUpdate2[0].length; k++) {
                        weightsInUpdate2[i][k] = 0;
                        weightsForgetUpdate2[i][k] = 0;
                        weightsMemoryUpdate2[i][k] = 0;
                        weightsOutUpdate2[i][k] = 0;
                    }
                }

                
                
                if (counter % numHiddenIters == 0) {
                    counter= 0;
                    DecimalFormat df = new DecimalFormat(".00");
                    
                    System.out.print("Pred: [");
                    for (int i = 0; i < numOfReviews; i++) {
                        System.out.print(df.format(outputs[i]) + " ");
                    }
                    System.out.println("]");

                    System.out.print("Actu: [");
                    for (int i = 0; i < numOfReviews; i++) {
                        System.out.print(y_train[i] + "   ");
                    }
                    System.out.println("]");

//                    System.out.print("Err : [");
//                    for (int i = 0; i < numOfReviews; i++) {
//                        System.out.print(y_train[i] + " ");
//                    }
//                    System.out.println("]");
                    
                    System.out.println("Accuracy: " + df.format(getPercentage(y_train, d, numOfReviews)) + " %");
                }

            }

        }
    }

    public static double getPercentage(int[] arrayA, double[] arrayB, int numOfReviews) {
        double percentage = 0;
        if(arrayA.length < numOfReviews) numOfReviews = arrayA.length;
        if(arrayB.length < numOfReviews) numOfReviews = arrayB.length;
        for (int i = 0; i < numOfReviews; i++) {
            if (arrayA[i] == (int) arrayB[i]) {
                /*note the different indices*/
                percentage++;
                /*count how many times you have matching values*/
 /* NOTE: This only works if you don't have repeating values in arrayA*/
            }

        }

        return (percentage / numOfReviews) * 100;
        /*return the amount of times over the length times 100*/
    }

}
