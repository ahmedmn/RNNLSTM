package rnnlstm;
import java.util.BitSet;
import java.util.Random;


public class BinaryAdding {
	

//	# compute sigmoid nonlinearity
//	def sigmoid(x):
//	    output = 1/(1+np.exp(-x))
//	    return output

	public static double computeSigmoid(double x) {
		return 1/(1+ Math.exp(-x));
	}
	
//	# convert output of sigmoid function to its derivative
//	def sigmoid_output_to_derivative(output):
//	    return output*(1-output)
//
	public static double sigmoidOutputToDerivative(double x) {
		return x*(1-x);
	}
	
	public static BitSet getBitSet(int value, int dim) {
	    BitSet bits = new BitSet();
	    int index = 0;
	    while (value != 0L) {
	    if (value % 2L != 0) {
	        bits.set(dim - index -1);
	      }
	      ++index;
	      value = value >>> 1;
	    }
	    return bits;
	}	
	
	
	public static double[] vectorMatrixMult(double [] vect, double [][] matrix) {
		if(vect.length != matrix.length) 
			throw new IllegalArgumentException("Dimension mismatch, vector length is " 
												+ vect.length + " and matrix column length is " + matrix.length + ".");
		double[] result = new double[matrix[0].length];
		for(int i=0;i<matrix[0].length;i++) {
			result[i] = 0;
			for(int j =0;j<vect.length;j++) {
				result[i] += vect[j] * matrix[j][i];
			}
		}
		return result;
	}

	public static double vectorVectorMultDot(double [] vect1, double [] vect2) {
		if(vect1.length != vect2.length) 
			throw new IllegalArgumentException("Dimension mismatch, vector lengths are " 
					+ vect1.length + " and " + vect2.length + ".");		
		double result = 0;
		for(int i=0;i<vect1.length;i++) {
			result += vect1[i] * vect2[i];
		}
		return result;
	}
	
	public static double[] vectorVectorMultAsterisk(double [] vect1, double [] vect2) {
		if(vect1.length != vect2.length) 
			throw new IllegalArgumentException("Dimension mismatch, vector lengths are " 
					+ vect1.length + " and " + vect2.length + ".");		
		double[] result = new double[vect1.length];
		for(int i=0;i<vect1.length;i++) {
			result[i] = vect1[i] * vect2[i];
		}
		return result;
	}
	
	public static double[][] vectorVectorMultDotM(double [] vect1, double [] vect2) {
		double[][] result = new double[vect1.length][];
		for(int i=0;i<vect1.length;i++) {
			result[i] = new double[vect2.length];
			for(int j=0;j<vect2.length;j++){
				result[i][j] = vect1[i] * vect2[j];
			}
		}
		return result;
	}
	
	public static void addVectors(double[] vect1, double[] vect2) {
		if(vect1.length != vect2.length) 
			throw new IllegalArgumentException("Dimension mismatch, vector lengths are " 
												+ vect1.length + " and " + vect2.length + ".");		
		for(int i=0;i<vect1.length; i++) {
			vect1[i] += vect2[i]; 
		}
	}
	
	public static void addMatrices(double[][] matrix1, double[][] matrix2) {
		if(matrix1.length != matrix2.length) 
			throw new IllegalArgumentException("Dimension mismatch, number of lines in matrices are " 
												+ matrix1.length + " and " + matrix2.length + ".");	
		if(matrix1[0].length != matrix2[0].length) 
			throw new IllegalArgumentException("Dimension mismatch, number of columns in matrices are " 
												+ matrix1[0].length + " and " + matrix2[0].length + ".");	

		for(int i=0;i<matrix1.length; i++) {
			for(int j=0;j<matrix1[0].length;j++)
				matrix1[i][j] += matrix2[i][j]; 
		}
	}
	
	public static double[] vectSigmoid(double [] vect) {
		double[] result = new double[vect.length];
		for(int i=0;i<vect.length; i++) {
			result[i] = computeSigmoid(vect[i]); 
		}
		return result;
	}

	public static double[] vectSigmoidOutputToDerivative(double[] vect) {
		double[] result = new double[vect.length];
		for(int i=0;i<vect.length; i++) {
			result[i] = sigmoidOutputToDerivative(vect[i]); 
		}
		return result;
	}
	
	public static double[] scalarVectMult(double scalar, double[] vect) {
		double[] result = new double [vect.length];
		for(int i=0;i<vect.length;i++) {
			result[i] = scalar * vect[i];
		}
		return result;
	}

	public static double[][] scalarMatrixMult(double scalar, double[][] matrix) {
		double[][] result = new double[matrix.length][];
		for(int i=0;i<matrix.length;i++) {
			result[i] = scalarVectMult(scalar, matrix[i]);
		}
		return result;
	}
	
	public static double[][] transpose(double[][] matrix) {
		double[][] result = new double[matrix[0].length][];
		for(int i = 0;i<matrix[0].length;i++) {
			result[i] = new double[matrix.length];
			
			for(int j=0;j<matrix.length;j++) {
				result[i][j] = matrix[j][i];
			}
		}
		return result;
	}

	private static String toString(double[] d) {
		String output= "[";
		for(int i=0;i<d.length-1;i++) {
			output += d[i] + ", ";
		}
		output += d[d.length-1] + "]";
		return output;
	}

	private static String toString(BitSet d, int numDigits) {
		String output= "[";
		for(int i=0;i<numDigits-1;i++) {
			output += (d.get(i)?1.0:0.0) + ", ";
		}
		output += (d.get(numDigits-1)?1.0:0.0) + "]";
		return output;
	}
	

	
	public static void main(String[] args) {
		
//		import copy, numpy as np
//		np.random.seed(0)
//
//
//
//		# training dataset generation
//		binary_dim = 8
//
//		largest_number = pow(2,binary_dim)

		int binaryDim = 8;
		int largestNumber = (int)Math.pow(2,binaryDim);
		
//		# input variables
//		alpha = 0.1
//		input_dim = 2
//		hidden_dim = 16
//
		double alpha = 0.1;
		int inputDim = 2;
		int hiddenDim = 16;
		
//
//		# initialize neural network weights
//		synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
//		synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
//		synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1
//		
//		synapse_0_update = np.zeros_like(synapse_0)
//		synapse_1_update = np.zeros_like(synapse_1)
//		synapse_h_update = np.zeros_like(synapse_h)
//
		double[][] weightsIn = new double[inputDim][];
		double[][] weightsInUpdate = new double[inputDim][];
		for(int i=0;i<inputDim;i++) {
			weightsIn[i] = new double[hiddenDim];
			weightsInUpdate[i] = new double[hiddenDim];
			for(int j=0;j<hiddenDim;j++) {
				weightsIn[i][j] = Math.random() * 2 - 1;
				weightsInUpdate[i][j] = 0;
			}
		}
		
		double[] weightsOut = new double[hiddenDim];
		double[] weightsOutUpdate = new double[hiddenDim];
		for(int i=0;i<hiddenDim;i++) {
			weightsOut[i] = Math.random() * 2 - 1;
			weightsOutUpdate[i] = 0;
		}
		
		double[][] weightsHidden = new double[hiddenDim][];
		double[][] weightsHiddenUpdate = new double[hiddenDim][];
		for(int i=0;i<hiddenDim;i++) {
			weightsHidden[i] = new double[hiddenDim];
			weightsHiddenUpdate[i] = new double[hiddenDim];
			for(int j=0;j<hiddenDim;j++) {
				weightsHidden[i][j] = Math.random() * 2 - 1;
				weightsHiddenUpdate[i][j] = 0;
			}
		}

		
//		# training logic
//		for j in range(10000):
//		    
//		    # generate a simple addition problem (a + b = c)
//		    a_int = np.random.randint(largest_number/2) # int version
//		    a = int2binary[a_int] # binary encoding
//
//		    b_int = np.random.randint(largest_number/2) # int version
//		    b = int2binary[b_int] # binary encoding
//
//		    # true answer
//		    c_int = a_int + b_int
//		    c = int2binary[c_int]
//		    
//		    # where we'll store our best guess (binary encoded)
//		    d = np.zeros_like(c)
//
//		    overallError = 0
//		    
//		    layer_2_deltas = list()
//		    layer_1_values = list()
//		    layer_1_values.append(np.zeros(hidden_dim))
		
		Random rand = new Random();
		
		for(int j=0;j<1000000;j++) {

			int aInt = rand.nextInt(largestNumber/2);
//			aInt = 7;
			BitSet a = getBitSet(aInt,binaryDim);
			
			int bInt = rand.nextInt(largestNumber/2);
//			bInt = 9;
			BitSet b = getBitSet(bInt,binaryDim);
			
			int cInt = aInt + bInt;
			BitSet c = getBitSet(cInt,binaryDim);
			
			double[] d = new double[binaryDim];
			
			double overallError = 0;
			
			double[] layer2Deltas = new double[binaryDim+1]; 
			double[][] layer1Values = new double[binaryDim+1][];
			layer1Values[0] = new double[hiddenDim];
			for(int k=0;k<hiddenDim;k++) {
				layer1Values[0][k]=0;
			}

		
//		    # moving along the positions in the binary encoding
//		    for position in range(binary_dim):
//		        
//		        # generate input and output
//		        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
//		        y = np.array([[c[binary_dim - position - 1]]]).T
//
//		        # hidden layer (input ~+ prev_hidden)
//		        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
//
//		        # output layer (new binary representation)
//		        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
//
//		        # did we miss?... if so, by how much?
//		        layer_2_error = y - layer_2
//		        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
//		        overallError += np.abs(layer_2_error[0])
//		    
//		        # decode estimate so we can print it out
//		        d[binary_dim - position - 1] = np.round(layer_2[0][0])
//		        
//		        # store hidden layer so we can use it in the next timestep
//		        layer_1_values.append(copy.deepcopy(layer_1))
//		    
//		    future_layer_1_delta = np.zeros(hidden_dim)
//		    

			
			double[] layer1, x, temp;
			double layer2, y, layer2Error;  
			for(int position = 0;position < binaryDim;position++) {

				x = new double[2];
				x[0]= a.get(binaryDim - position - 1)?1:0;
				x[1]= b.get(binaryDim - position - 1)?1:0;

				y = c.get(binaryDim - position - 1)?1:0;
				
				temp = vectorMatrixMult(x,weightsIn);
				addVectors(temp, vectorMatrixMult(layer1Values[position],weightsHidden));
				layer1 = vectSigmoid(temp);
				
				layer2 = computeSigmoid(vectorVectorMultDot(layer1, weightsOut)); 
				
				layer2Error = y - layer2;
				layer2Deltas[position+1] = layer2Error * sigmoidOutputToDerivative(layer2);
				overallError += Math.abs(layer2Error);
				
				d[binaryDim - position - 1] = Math.round(layer2);
				
				layer1Values[position+1] = layer1;
			}
			
			double[] futureLayer1Delta = new double[hiddenDim];
			for(int i=0;i<futureLayer1Delta.length;i++) {
				futureLayer1Delta[i] = 0;
			}
			
//		    for position in range(binary_dim):
//		        
//		        X = np.array([[a[position],b[position]]])
//		        layer_1 = layer_1_values[-position-1]
//		        prev_layer_1 = layer_1_values[-position-2]
//		        
//		        # error at output layer
//		        layer_2_delta = layer_2_deltas[-position-1]
//		        # error at hidden layer
//		        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
//
//		        # let's update all our weights so we can try again
//		        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
//		        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
//		        synapse_0_update += X.T.dot(layer_1_delta)
//		        
//		        future_layer_1_delta = layer_1_delta
//		    

			double[] prevLayer1, layer1Delta;
			double layer2Delta;
			for(int position = binaryDim;position>0;position--) {
				x = new double[2];
				x[0]= a.get(binaryDim - position)?1:0;
				x[1]= b.get(binaryDim - position)?1:0;
				layer1 = layer1Values[position];
				prevLayer1 = layer1Values[position-1];
				
				layer2Delta = layer2Deltas[position];
				
				temp = vectorMatrixMult(futureLayer1Delta, transpose(weightsHidden));
				addVectors(temp, scalarVectMult(layer2Delta,weightsOut));
				layer1Delta = vectorVectorMultAsterisk(temp,vectSigmoidOutputToDerivative(layer1));
				
				addVectors(weightsOutUpdate, scalarVectMult(layer2Delta, layer1));
				addMatrices(weightsHiddenUpdate, vectorVectorMultDotM(prevLayer1, layer1Delta));
				addMatrices(weightsInUpdate, vectorVectorMultDotM(x,layer1Delta));

				futureLayer1Delta = layer1Delta;
			}
			
			
//
//		    synapse_0 += synapse_0_update * alpha
//		    synapse_1 += synapse_1_update * alpha
//		    synapse_h += synapse_h_update * alpha    
//
//		    synapse_0_update *= 0
//		    synapse_1_update *= 0
//		    synapse_h_update *= 0
//		    
			addVectors(weightsOut, scalarVectMult(alpha, weightsOutUpdate));
			addMatrices(weightsHidden, scalarMatrixMult(alpha, weightsHiddenUpdate));
			addMatrices(weightsIn, scalarMatrixMult(alpha, weightsInUpdate));

			for(int i=0;i<weightsOutUpdate.length;i++){
				weightsOutUpdate[i]=0;
			}
			for(int i=0;i<weightsHiddenUpdate.length;i++){
				for(int k=0;k<weightsHiddenUpdate[0].length;k++)
					weightsHiddenUpdate[i][k]=0;
			}
			for(int i=0;i<weightsInUpdate.length;i++){
				for(int k=0;k<weightsInUpdate[0].length;k++)
					weightsInUpdate[i][k]=0;
			}
			
			
//		    # print out progress
//		    if(j % 1000 == 0):
//		        print "Error:" + str(overallError)
//		        print "Pred:" + str(d)
//		        print "True:" + str(c)
//		        out = 0
//		        for index,x in enumerate(reversed(d)):
//		            out += x*pow(2,index)
//		        print str(a_int) + " + " + str(b_int) + " = " + str(out)
//		        print "------------"
			
			if(j % 1000== 0){
				System.out.println("Error: " + overallError);
				System.out.println("Pred: " + toString(d));
				System.out.println("True: " + toString(c, binaryDim));
				int out = 0;
				for(int i =0; i<binaryDim;i++) {
					out += d[binaryDim-i-1]*((int)Math.pow(2, i));
				}
				System.out.print(aInt + " + " + bInt + " = " + out);
				if (out != cInt) System.out.println("\t\t\t\t\tWRONG");
				System.out.println("\n------------");
			}

		}    

		
		
	}



}
