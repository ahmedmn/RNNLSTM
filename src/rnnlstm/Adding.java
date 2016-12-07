//package rnnlstm;
//import java.util.BitSet;
//import java.util.Random;
//
//public class Adding
//{   
//    
//    
//    
//    public static void main(String[] args) {
//        
//        Matrix matrix = new Matrix();
//        BitSetOperations bitSet = new BitSetOperations();
//        
//        int binaryDim = 8;
//        int largestNumber = (int)Math.pow(2,binaryDim);
//        
//        
//        double alpha = 0.1;
//        int inputDim = 2;
//        int hiddenDim = 8;
//        
//        
//        double[][] weightsIn = new double[inputDim][];
//        double[][] weightsInUpdate = new double[inputDim][];
//        for(int i=0;i<inputDim;i++) {
//            weightsIn[i] = new double[hiddenDim];
//            weightsInUpdate[i] = new double[hiddenDim];
//            for(int j=0;j<hiddenDim;j++) {
//                weightsIn[i][j] = Math.random() * 2 - 1;
//                weightsInUpdate[i][j] = 0;
//            }
//        }
//        
//        double[] weightsOut = new double[hiddenDim];
//        double[] weightsOutUpdate = new double[hiddenDim];
//        for(int i=0;i<hiddenDim;i++) {
//            weightsOut[i] = Math.random() * 2 - 1;
//            weightsOutUpdate[i] = 0;
//        }
//        
//        double[][] weightsHidden = new double[hiddenDim][];
//        double[][] weightsHiddenUpdate = new double[hiddenDim][];
//        for(int i=0;i<hiddenDim;i++) {
//            weightsHidden[i] = new double[hiddenDim];
//            weightsHiddenUpdate[i] = new double[hiddenDim];
//            for(int j=0;j<hiddenDim;j++) {
//                weightsHidden[i][j] = Math.random() * 2 - 1;
//                weightsHiddenUpdate[i][j] = 0;
//            }
//        }
//        
//        Random rand = new Random();
//        
//        for(int j=0;j<1000000;j++) {
//
//            int aInt = rand.nextInt(largestNumber/2);
//            BitSet a = bitSet.getBitSet(aInt,binaryDim);
//            
//            int bInt = rand.nextInt(largestNumber/2);
//            BitSet b = bitSet.getBitSet(bInt,binaryDim);
//            
//            int cInt = aInt + bInt;
//            BitSet c = bitSet.getBitSet(cInt,binaryDim);
//            
//            double[] d = new double[binaryDim];
//            
//            double overallError = 0;
//            
//            double[] layer2Deltas = new double[binaryDim+1]; 
//            double[][] layer1Values = new double[binaryDim+1][];
//            layer1Values[0] = new double[hiddenDim];
//            for(int k=0;k<hiddenDim;k++) {
//                layer1Values[0][k]=0;
//            }
//
//            
//            double[] layer1, x, temp;
//            double layer2, y, layer2Error;  
//            for(int position = 0;position < binaryDim;position++) {
//
//                x = new double[2];
//                x[0]= a.get(binaryDim - position - 1)?1:0;
//                x[1]= b.get(binaryDim - position - 1)?1:0;
//
//                y = c.get(binaryDim - position - 1)?1:0;
//                
//                temp = matrix.vectorMatrixMult(x,weightsIn);
//                matrix.addVectorsUpdate(temp, matrix.vectorMatrixMult(layer1Values[position],weightsHidden));
//                layer1 = matrix.vectSigmoid(temp);
//                
//                layer2 = matrix.computeSigmoid(matrix.vectorVectorMultDot(layer1, weightsOut)); 
//                
//                layer2Error = y - layer2;
//                layer2Deltas[position+1] = layer2Error * matrix.sigmoidOutputToDerivative(layer2);
//                overallError += Math.abs(layer2Error);
//                
//                d[binaryDim - position - 1] = Math.round(layer2);
//                
//                layer1Values[position+1] = layer1;
//            }
//            
//            double[] futureLayer1Delta = new double[hiddenDim];
//            for(int i=0;i<futureLayer1Delta.length;i++) {
//                futureLayer1Delta[i] = 0;
//            } 
//
//            double[] prevLayer1, layer1Delta;
//            double layer2Delta;
//            for(int position = binaryDim;position>0;position--) {
//                x = new double[2];
//                x[0]= a.get(binaryDim - position)?1:0;
//                x[1]= b.get(binaryDim - position)?1:0;
//                layer1 = layer1Values[position];
//                prevLayer1 = layer1Values[position-1];
//                
//                layer2Delta = layer2Deltas[position];
//                
//                temp = matrix.vectorMatrixMult(futureLayer1Delta, matrix.transpose(weightsHidden));
//                matrix.addVectorsUpdate(temp, matrix.scalarVectMult(layer2Delta,weightsOut));
//                layer1Delta = matrix.vectorVectorMultAsterisk(temp,matrix.vectSigmoidOutputToDerivative(layer1));
//                
//                matrix.addVectorsUpdate(weightsOutUpdate, matrix.scalarVectMult(layer2Delta, layer1));
//                matrix.addMatricesUpdate(weightsHiddenUpdate, matrix.vectorVectorMultDotM(prevLayer1, layer1Delta));
//                matrix.addMatricesUpdate(weightsInUpdate, matrix.vectorVectorMultDotM(x,layer1Delta));
//
//                futureLayer1Delta = layer1Delta;
//            }
//            
//                      
//            matrix.addVectorsUpdate(weightsOut, matrix.scalarVectMult(alpha, weightsOutUpdate));
//            matrix.addMatricesUpdate(weightsHidden, matrix.scalarMatrixMult(alpha, weightsHiddenUpdate));
//            matrix.addMatricesUpdate(weightsIn, matrix.scalarMatrixMult(alpha, weightsInUpdate));
//
//            for(int i=0;i<weightsOutUpdate.length;i++){
//                weightsOutUpdate[i]=0;
//            }
//            for(int i=0;i<weightsHiddenUpdate.length;i++){
//                for(int k=0;k<weightsHiddenUpdate[0].length;k++)
//                    weightsHiddenUpdate[i][k]=0;
//            }
//            for(int i=0;i<weightsInUpdate.length;i++){
//                for(int k=0;k<weightsInUpdate[0].length;k++)
//                    weightsInUpdate[i][k]=0;
//            }
//            
//            
//            
//           if(j % 1000== 0){
//                System.out.println("Error: " + overallError);
//                System.out.println("Pred: " + bitSet.toString(d));
//                System.out.println("True: " + bitSet.toString(c, binaryDim));
//                int out = 0;
//                for(int i =0; i<binaryDim;i++) {
//                    out += d[binaryDim-i-1]*((int)Math.pow(2, i));
//                }
//                System.out.print(aInt + " + " + bInt + " = " + out);
//                if (out != cInt) System.out.println("\t\t\t\t\tWRONG");
//                System.out.println("\n------------");
//            }
//
//        }    
//
//        
//        
//    }
//}
//
