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

        List<String> X_trainStr = Files.readAllLines(Paths.get(args[0]));
        List<String> y_trainStr = Files.readAllLines(Paths.get(args[1]));

        List<String> X_testStr = Files.readAllLines(Paths.get(args[2]));
        List<String> y_testStr = Files.readAllLines(Paths.get(args[3]));

        int[][] X_train = LoadData(X_trainStr);
        int[] y_train = Arrays.asList(y_trainStr.toArray(new String[y_trainStr.size()])).stream().mapToInt(Integer::parseInt).toArray();
        int[][] X_test = LoadData(X_testStr);
        int[] y_test = Arrays.asList(y_testStr.toArray(new String[y_testStr.size()])).stream().mapToInt(Integer::parseInt).toArray();

        System.out.println("Training Sample 0");
        System.out.print("[");
        for (int i = 0; i < X_train[0].length; i++) {
            System.out.print(X_train[0][i] + " ");
        }
        System.out.println("]" + "\n" + y_train[0]);

        System.out.println("Training Sample 1");
        System.out.print("[");
        for (int i = 0; i < X_train[1].length; i++) {
            System.out.print(X_train[1][i] + " ");
        }
        System.out.println("]" + "\n" + y_train[1]);

//        List<ArrayList> X_train = new ArrayList<ArrayList>(Arrays.asList(X_trainStr.split))
        Sigmoid sigmoid = new Sigmoid();
        Matrix matrix = new Matrix();
        Vector vector = new Vector();

        ////parameters of neural network
        
        double alpha = 0.05;
        int hiddenDim = 200;
        int numOfReviews = 50;
        int numOfIterations = 1000000;
        int numNotPrintedIters = numOfReviews;
        
        int dataRowNum = X_train.length;
        int dataColNum = X_train[0].length;
        double[] d          = new double[dataRowNum];        
        double[] d2         = new double[dataRowNum];
        double[] outputs    = new double[dataRowNum];
        double[] errors     = new double[dataRowNum];
        
        //weights for input and corresponding gates
        double[] weightsIn          = new double[hiddenDim];
        double[] weightsInUpdate    = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsIn[i] = Math.random() * 2 - 1;
            weightsInUpdate[i] = 0;
        }

        double[] weightsForget          = new double[hiddenDim];
        double[] weightsForgetUpdate    = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsForget[i] = Math.random() * 2 - 1;
            weightsForgetUpdate[i] = 0;
        }

        double[] weightsMemory          = new double[hiddenDim];
        double[] weightsMemoryUpdate    = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsMemory[i] = Math.random() * 2 - 1;
            weightsMemoryUpdate[i] = 0;
        }

        double[] weightsOut         = new double[hiddenDim];
        double[] weightsOutUpdate   = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsOut[i] = Math.random() * 2 - 1;
            weightsOutUpdate[i] = 0;
        }

        //final output weights
        double[] weightsFinal = new double[hiddenDim];
        double[] weightsFinalUpdate = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            weightsFinal[i] = Math.random() * 2 - 1;
            weightsFinalUpdate[i] = 0;
        }
        
        // weitghts for output of previous stage and corresponding gates
        double[] temp;
        double[] temp2;

        double[][] weightsIn2       = new double[hiddenDim][];
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

        double[][] weightsForget2       = new double[hiddenDim][];
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

        double[][] weightsMemory2       = new double[hiddenDim][];
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

        double[][] weightsOut2          = new double[hiddenDim][];
        double[][] weightsOutUpdate2    = new double[hiddenDim][];
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



        double[] x;
        double[][] forgetGate, inputGate, outputGate, memoryInput, memory, output, almostOutput;
        forgetGate = new double[dataColNum + 1][]; //f^t
        inputGate = new double[dataColNum + 1][]; //i^t
        outputGate = new double[dataColNum + 1][]; //o^t
        memoryInput = new double[dataColNum + 1][]; //a^t
        memory = new double[dataColNum + 1][]; //c^t
        output = new double[dataColNum + 1][]; //h^t
        almostOutput = new double[dataColNum + 1][];
        double finalOutput, y, finalOutputError, finalOutputDelta;        
        
        //initialization
        for(int j=1;j<dataColNum + 1;j++) {
            forgetGate[j] = new double[hiddenDim];
            inputGate[j] = new double[hiddenDim];
            outputGate[j] = new double[hiddenDim];
            memoryInput[j] = new double[hiddenDim];
            memory[j] = new double[hiddenDim];
            output[j] = new double[hiddenDim];
            almostOutput[j] = new double[hiddenDim];
        }
        memory[0] = new double[hiddenDim];
        output[0] = new double[hiddenDim];
        
        double[] futureOutputDelta, currentOutputDelta, futureMemoryDelta, currentMemoryDelta, aux;
        futureOutputDelta = new double[hiddenDim];
        futureMemoryDelta = new double[hiddenDim];
        currentOutputDelta = new double[hiddenDim];
        currentMemoryDelta = new double[hiddenDim];        
        
        temp = new double[hiddenDim];
        temp2 = new double[hiddenDim];
        double[][] tempMat = new double[hiddenDim][];
        for(int i=0; i<tempMat.length; i++){
            tempMat[i] = new double[hiddenDim];
        }
        
        int i, j, k, reviewRow, reviewRow2, wordCol, counter =0;
        double[] temp3, temp4, temp5, temp6;

        for (k = 0; k < hiddenDim; k++) {
            memory[0][k] = 0;
            output[0][k] = 0;
        }

        for (k = 0; k < dataColNum+1; k++) {
            almostOutput[k] = new double[hiddenDim];
        }

        x = new double[dataColNum + 1];
        
        for (j = 0; j < numOfIterations; j++) {


            for (reviewRow = 0; reviewRow < numOfReviews; reviewRow++) { //TODO erase
//            for (int reviewRow = 0; reviewRow < dataRowNum; reviewRow++) {
                counter++;

                for (wordCol = 0; wordCol < dataColNum; wordCol++) {
                    x[wordCol+1] = X_train[reviewRow][wordCol];
                }

                y = y_train[reviewRow];

                //forward pass
                for (wordCol = 1; wordCol <= dataColNum; wordCol++) {
                    vector.scalarVectMultNoOut(forgetGate[wordCol], x[wordCol], weightsForget);
                    matrix.vectorMatrixMultNoOutAdd(forgetGate[wordCol], output[wordCol-1], weightsForget2);
                    vector.vectSigmoidNoOut(forgetGate[wordCol]);
                    vector.vectorVectorMultAsteriskNoOut(memory[wordCol], memory[wordCol-1], forgetGate[wordCol]);

                    vector.scalarVectMultNoOut(inputGate[wordCol], x[wordCol], weightsIn);
                    matrix.vectorMatrixMultNoOutAdd(inputGate[wordCol], output[wordCol-1], weightsIn2);
                    vector.vectSigmoidNoOut(inputGate[wordCol]);

                    vector.scalarVectMultNoOut(memoryInput[wordCol], x[wordCol], weightsMemory);
                    matrix.vectorMatrixMultNoOutAdd(memoryInput[wordCol], output[wordCol-1], weightsMemory2);
                    vector.vectTangentHNoOut(memoryInput[wordCol]);

                    vector.vectorVectorMultAsteriskNoOutAdd(memory[wordCol], 
                                                            inputGate[wordCol],
                                                            memoryInput[wordCol]);

                    vector.copy(almostOutput[wordCol], memory[wordCol]);
                    vector.vectTangentHNoOut(almostOutput[wordCol]);

                    vector.scalarVectMultNoOut(outputGate[wordCol], x[wordCol], weightsOut);
                    matrix.vectorMatrixMultNoOutAdd(outputGate[wordCol], output[wordCol-1], weightsOut2);
                    vector.vectSigmoidNoOut(outputGate[wordCol]);

                    vector.vectorVectorMultAsteriskNoOut(output[wordCol], outputGate[wordCol], almostOutput[wordCol]);
                }
                
                //compute output
                finalOutput = sigmoid.computeSigmoid(vector.vectorVectorMultDot(output[dataColNum], weightsFinal));
                finalOutputError = 2*(y-finalOutput); //Math.pow((y - finalOutput),2)                
                d[reviewRow] = Math.round(finalOutput);
                outputs[reviewRow]  = finalOutput;
                errors[reviewRow]   = finalOutputError;
                finalOutputDelta = (finalOutputError * sigmoid.sigmoidOutputToDerivative(finalOutput));

                vector.scalarVectMultNoOut(weightsFinalUpdate, finalOutputDelta, weightsFinal); //computing dh

                vector.copy(futureOutputDelta, weightsFinalUpdate);
                for (k = 0; k < hiddenDim; k++) {
                    futureMemoryDelta[k] = 0;
                }

                //backpropagation
                for (wordCol = dataColNum; wordCol > 0; wordCol--) {

                    for (k = 0; k < hiddenDim; k++) {
                        currentOutputDelta[k] = 0;
                        currentMemoryDelta[k] = 0;
                    }
                    

                    vector.vectorVectorMultAsteriskNoOut(temp, futureOutputDelta, almostOutput[wordCol]); //do^t
                    vector.vectSigmoidOutputToDerivativeNoOut(temp2, outputGate[wordCol]);
                    vector.vectorVectorMultAsteriskNoOut(temp, temp, temp2);//d(o^)^t
                    vector.scalarVectMultNoOutAdd(weightsOutUpdate, x[wordCol], temp);
                    vector.vectorVectorMultDotMNoOutAdd(weightsOutUpdate2, output[wordCol-1], temp);
                    //computing dh^(t-1)
                    matrix.vectorTransposeMatrixMultNoOutAdd(currentOutputDelta, temp, weightsOut2);

//                    temp = vector.vectorVectorMultAsterisk(futureOutputDelta, almostOutput[wordCol]); //do^t
//                    temp = vector.vectorVectorMultAsterisk(temp, vector.vectSigmoidOutputToDerivative(outputGate[wordCol]));//d(o^)^t
//                    vector.addVectors(weightsOutUpdate, vector.scalarVectMult(x[wordCol], temp)); //TODO check whether it is correct
//                    matrix.addMatrices(weightsOutUpdate2, vector.vectorVectorMultDotM(output[wordCol-1], temp));
//                    //computing dh^(t-1)
//                    currentOutputDelta = matrix.vectorMatrixMult(temp, matrix.transpose(weightsOut2));

                    vector.vectorVectorMultAsteriskNoOut(temp, futureOutputDelta, outputGate[wordCol]); // dh^t . o^t
                    vector.vectTangentHToDerivativeNoOut(almostOutput[wordCol]);
                    vector.vectorVectorMultAsteriskNoOut(temp, temp, almostOutput[wordCol]);
                    vector.addVectors(futureMemoryDelta, temp); // dc^t
                    
//                    temp = vector.vectorVectorMultAsterisk(futureOutputDelta, outputGate[wordCol]); // dh^t . o^t
//                    vector.vectTangentHToDerivativeNoOut(almostOutput[wordCol]);
//                    temp = vector.vectorVectorMultAsterisk(temp, almostOutput[wordCol]);
//                    vector.addVectors(futureMemoryDelta, temp); // dc^t

                    vector.vectorVectorMultAsteriskNoOut(currentMemoryDelta, futureMemoryDelta, forgetGate[wordCol]); //dc^(t-1)

//                    currentMemoryDelta = vector.vectorVectorMultAsterisk(futureMemoryDelta, forgetGate[wordCol]); //dc^(t-1)

                    //forget gate
                    vector.vectorVectorMultAsteriskNoOut(temp, futureMemoryDelta, memory[wordCol - 1]); //df^t
                    vector.vectSigmoidOutputToDerivativeNoOut(temp2, forgetGate[wordCol]);
                    vector.vectorVectorMultAsteriskNoOut(temp, temp, temp2);
                    vector.scalarVectMultNoOutAdd(weightsForgetUpdate, x[wordCol], temp);
                    vector.vectorVectorMultDotMNoOutAdd(weightsForgetUpdate2, output[wordCol-1], temp);
                    //computing dh^(t-1)
                    matrix.vectorTransposeMatrixMultNoOutAdd(currentOutputDelta, temp, weightsForget2);


//                    //forget gate
//                    temp = vector.vectorVectorMultAsterisk(futureMemoryDelta, memory[wordCol - 1]); //df^t
//                    temp = vector.vectorVectorMultAsterisk(temp, vector.vectSigmoidOutputToDerivative(forgetGate[wordCol]));
//                    vector.addVectors(weightsForgetUpdate, vector.scalarVectMult(x[wordCol], temp)); //TODO check whether it is correct
//                    matrix.addMatrices(weightsForgetUpdate2, vector.vectorVectorMultDotM(output[wordCol-1], temp));
//                    //computing dh^(t-1)
//                    vector.addVectors(currentOutputDelta, matrix.vectorMatrixMult(temp, matrix.transpose(weightsForget2)));

                    //input gate
                    vector.vectorVectorMultAsteriskNoOut(temp, futureMemoryDelta, memoryInput[wordCol]); //di^t
                    vector.vectSigmoidOutputToDerivativeNoOut(temp2, inputGate[wordCol]);
                    vector.vectorVectorMultAsteriskNoOut(temp, temp, temp2);
                    vector.scalarVectMultNoOutAdd(weightsInUpdate, x[wordCol], temp);
                    vector.vectorVectorMultDotMNoOutAdd(weightsInUpdate2,output[wordCol-1], temp);
                    //computing dh^(t-1)
                    matrix.vectorTransposeMatrixMultNoOutAdd(currentOutputDelta, temp, weightsIn2);

                    
//                    //input gate
//                    temp = vector.vectorVectorMultAsterisk(futureMemoryDelta, memoryInput[wordCol]); //di^t
//                    temp = vector.vectorVectorMultAsterisk(temp, vector.vectSigmoidOutputToDerivative(inputGate[wordCol]));
//                    vector.addVectors(weightsInUpdate, vector.scalarVectMult(x[wordCol], temp)); //TODO check whether it is correct
//                    matrix.addMatrices(weightsInUpdate2, vector.vectorVectorMultDotM(output[wordCol-1], temp));
//                    //computing dh^(t-1)
//                    vector.addVectors(currentOutputDelta, matrix.vectorMatrixMult(temp, matrix.transpose(weightsIn2)));                    
                    
                    //memory input
                    vector.vectorVectorMultAsteriskNoOut(temp, futureMemoryDelta, inputGate[wordCol]); //da^t
                    vector.vectTangentHToDerivativeNoOut(memoryInput[wordCol]);
                    vector.vectorVectorMultAsteriskNoOut(temp, temp, memoryInput[wordCol]); //d(a^)^t
                    vector.scalarVectMultNoOutAdd(weightsMemoryUpdate, x[wordCol], temp);
                    vector.vectorVectorMultDotMNoOutAdd(weightsMemoryUpdate2, output[wordCol-1], temp);
                    //computing dh^(t-1)
                    matrix.vectorTransposeMatrixMultNoOutAdd(currentOutputDelta, temp, weightsMemory2);

               
//                    //memory input
//                    temp = vector.vectorVectorMultAsterisk(futureMemoryDelta, inputGate[wordCol]); //da^t
//                    vector.vectTangentHToDerivativeNoOut(memoryInput[wordCol]);
//                    temp = vector.vectorVectorMultAsterisk(temp, memoryInput[wordCol]); //d(a^)^t
//                    vector.addVectors(weightsMemoryUpdate, vector.scalarVectMult(x[wordCol], temp)); //TODO check whether it is correct
//                    matrix.addMatrices(weightsMemoryUpdate2, vector.vectorVectorMultDotM(output[wordCol-1], temp));
//                    //computing dh^(t-1)
//                    vector.addVectors(currentOutputDelta, matrix.vectorMatrixMult(temp, matrix.transpose(weightsMemory2)));                    
                    
                    // preparing new values of memory and output
                    aux = futureMemoryDelta;
                    futureMemoryDelta = currentMemoryDelta;
                    currentMemoryDelta = aux;
                    aux = futureOutputDelta;
                    futureOutputDelta = currentOutputDelta;
                    currentOutputDelta = aux;

                }

                
                vector.scalarVectMultNoOutAdd(weightsIn,    alpha,   weightsInUpdate);
                vector.scalarVectMultNoOutAdd(weightsForget,alpha,   weightsForgetUpdate);                
                vector.scalarVectMultNoOutAdd(weightsMemory,alpha,   weightsMemoryUpdate);
                vector.scalarVectMultNoOutAdd(weightsOut,   alpha,   weightsOutUpdate);
                vector.scalarVectMultNoOutAdd(weightsFinal, alpha,   weightsFinalUpdate);
                
                
                matrix.scalarMatrixMultNoOutAdd(weightsIn2,     alpha, weightsInUpdate2);
                matrix.scalarMatrixMultNoOutAdd(weightsForget2, alpha, weightsForgetUpdate2);                
                matrix.scalarMatrixMultNoOutAdd(weightsMemory2, alpha, weightsMemoryUpdate2);
                matrix.scalarMatrixMultNoOutAdd(weightsOut2,    alpha, weightsOutUpdate2);

                for (i = 0; i < weightsInUpdate.length; i++) {
                    weightsInUpdate[i]      = 0;
                    weightsForgetUpdate[i]  = 0;
                    weightsMemoryUpdate[i]  = 0;
                    weightsOutUpdate[i]     = 0;
                    weightsFinalUpdate[i]   = 0;
                }

                for (i = 0; i < weightsInUpdate2.length; i++) {
                    temp3 = weightsInUpdate2[i];
                    temp4 = weightsForgetUpdate2[i];
                    temp5 = weightsMemoryUpdate2[i];
                    temp6 = weightsOutUpdate2[i];
                    for (k = 0; k < weightsInUpdate2[0].length; k++) {
                        temp3[k] = 0;
                        temp4[k] = 0;
                        temp5[k] = 0;
                        temp6[k] = 0;
                    }
                }

                if (counter % 100 == 0) System.out.print(", " + counter);
                
                if (counter % numNotPrintedIters == 0) {
                    counter= 0;
                    DecimalFormat df = new DecimalFormat(".00");

                    System.out.println("================ " + j);

                    int rew = 20;
                    if(rew > numOfReviews) rew = numOfReviews;
                    
                    System.out.print("Pred: [");
                    for (i = 0; i < rew; i++) {
                        System.out.print(df.format(outputs[i]) + " ");
                    }
                    System.out.println("]");

                    System.out.print("Actu: [");
                    for (i = 0; i < rew; i++) {
                        System.out.print(y_train[i] + "   ");
                    }
                    System.out.println("]");

                    System.out.print("Err : [");
                    for (i = 0; i < rew; i++) {
                        System.out.print(df.format(errors[i]) + "   ");
                    }
                    System.out.println("]");
                    
                    System.out.println("Accuracy train data: " + df.format(getPercentage(y_train, d, numOfReviews)) + " %");
                    
                    int numOfReviews2 = numOfReviews;
                    if (numOfReviews2 > X_test.length) numOfReviews2 = X_test.length;
                    
                    {
                        //forward pass
                        for (reviewRow2 = 0; reviewRow2 < numOfReviews2; reviewRow2++) { 

                            for (wordCol = 0; wordCol < dataColNum; wordCol++) {
                                x[wordCol+1] = X_test[reviewRow2][wordCol];
                            }

                            y = y_test[reviewRow2];

                            //forward pass
                            for (wordCol = 1; wordCol <= dataColNum; wordCol++) {
                                vector.scalarVectMultNoOut(forgetGate[wordCol], x[wordCol], weightsForget);
                                matrix.vectorMatrixMultNoOutAdd(forgetGate[wordCol], output[wordCol-1], weightsForget2);
                                vector.vectSigmoidNoOut(forgetGate[wordCol]);
                                vector.vectorVectorMultAsteriskNoOut(memory[wordCol], memory[wordCol-1], forgetGate[wordCol]);

                                vector.scalarVectMultNoOut(inputGate[wordCol], x[wordCol], weightsIn);
                                matrix.vectorMatrixMultNoOutAdd(inputGate[wordCol], output[wordCol-1], weightsIn2);
                                vector.vectSigmoidNoOut(inputGate[wordCol]);

                                vector.scalarVectMultNoOut(memoryInput[wordCol], x[wordCol], weightsMemory);
                                matrix.vectorMatrixMultNoOutAdd(memoryInput[wordCol], output[wordCol-1], weightsMemory2);
                                vector.vectTangentHNoOut(memoryInput[wordCol]);

                                vector.vectorVectorMultAsteriskNoOutAdd(memory[wordCol], 
                                                                        inputGate[wordCol],
                                                                        memoryInput[wordCol]);

                                vector.copy(almostOutput[wordCol], memory[wordCol]);
                                vector.vectTangentHNoOut(almostOutput[wordCol]);

                                vector.scalarVectMultNoOut(outputGate[wordCol], x[wordCol], weightsOut);
                                matrix.vectorMatrixMultNoOutAdd(outputGate[wordCol], output[wordCol-1], weightsOut2);
                                vector.vectSigmoidNoOut(outputGate[wordCol]);

                                vector.vectorVectorMultAsteriskNoOut(output[wordCol], outputGate[wordCol], almostOutput[wordCol]);
                            }

                            //compute output
                            finalOutput = sigmoid.computeSigmoid(vector.vectorVectorMultDot(output[dataColNum], weightsFinal));
                            finalOutputError = 2*(y-finalOutput); //Math.pow((y - finalOutput),2)                
                            d2[reviewRow] = Math.round(finalOutput);
//                            outputs[reviewRow]  = finalOutput;
//                            errors[reviewRow]   = finalOutputError;
                        }
                        //end of forward pass
                    }
                    
//                    System.out.print("Pred: [");
//                    for (i = 0; i < rew; i++) {
//                        System.out.print(df.format(outputs[i]) + " ");
//                    }
//                    System.out.println("]");

//                    System.out.print("Actu: [");
//                    for (i = 0; i < rew; i++) {
//                        System.out.print(y_test[i] + "   ");
//                    }
//                    System.out.println("]");

//                    System.out.print("Err : [");
//                    for (i = 0; i < rew; i++) {
//                        System.out.print(df.format(errors[i]) + "   ");
//                    }
//                    System.out.println("]");
                    
                    System.out.println("Accuracy test  data: " + df.format(getPercentage(y_test, d2, numOfReviews2)) + " %");
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
