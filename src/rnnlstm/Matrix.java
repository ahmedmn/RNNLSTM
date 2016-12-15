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
public class Matrix {

    Vector vector = new Vector();

    public Matrix() {
    }

    public double[] vectorMatrixMult(double[] vect, double[][] matrix) {
//        if (vect.length != matrix.length) {
//            throw new IllegalArgumentException("Dimension mismatch, vector length is "
//                    + vect.length + " and matrix column length is " + matrix.length + ".");
//        }
        double[] result = new double[matrix[0].length];
        for (int i = 0; i < matrix[0].length; i++) {
            result[i] = 0;
            for (int j = 0; j < vect.length; j++) {
                result[i] += vect[j] * matrix[j][i];
            }
        }
        return result;
    }

    public void vectorMatrixMultNoOut(double[] result, double[] vect, double[][] matrix) {
//        if (vect.length != matrix.length) {
//            throw new IllegalArgumentException("Dimension mismatch, vector length is "
//                    + vect.length + " and matrix column length is " + matrix.length + ".");
//        }


//        for (int i = 0; i < matrix[0].length; i++) {
//            result[i] = 0;
//            for (int j = 0; j < vect.length; j++) {
//                result[i] += vect[j] * matrix[j][i];
//            }
//        }
        
        for (int i = 0; i < matrix[0].length; i++) {
            result[i] = 0;
        }
        double[] temp;
        for (int j = 0; j < vect.length; j++) {
            temp = matrix[j];
            for(int i=0; i< matrix[j].length; i++)
                result[i] += vect[j] * temp[i];
        }
    }

    public void vectorMatrixMultNoOutAdd(double[] result, double[] vect, double[][] matrix) {
//        if (vect.length != matrix.length) {
//            throw new IllegalArgumentException("Dimension mismatch, vector length is "
//                    + vect.length + " and matrix column length is " + matrix.length + ".");
//        }

//        for (int i = 0; i < matrix[0].length; i++) {
//            for (int j = 0; j < vect.length; j++) {
//                result[i] += vect[j] * matrix[j][i];
//            }
//        }

        double[] temp;
        for (int j = 0; j < vect.length; j++) {
            temp = matrix[j];
            for(int i=0; i< matrix[j].length; i++)
                result[i] += vect[j] * temp[i];
        }
    }
    
    public void addMatrices(double[][] matrix1, double[][] matrix2) {
//        if (matrix1.length != matrix2.length) {
//            throw new IllegalArgumentException("Dimension mismatch, number of lines in matrices are "
//                    + matrix1.length + " and " + matrix2.length + ".");
//        }
//        if (matrix1[0].length != matrix2[0].length) {
//            throw new IllegalArgumentException("Dimension mismatch, number of columns in matrices are "
//                    + matrix1[0].length + " and " + matrix2[0].length + ".");
//        }

        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix1[0].length; j++) {
                matrix1[i][j] += matrix2[i][j];
            }
        }
    }

    public double[][] scalarMatrixMult(double scalar, double[][] matrix) {
        double[][] result = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = vector.scalarVectMult(scalar, matrix[i]);
        }
        return result;
    }

    public void scalarMatrixMultNoOut(double scalar, double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            vector.scalarVectMultNoOut(scalar, matrix[i]);
        }
    }
    
    public double[][] transpose(double[][] matrix) {
        double[][] result = new double[matrix[0].length][];
        for (int i = 0; i < matrix[0].length; i++) {
            result[i] = new double[matrix.length];

            for (int j = 0; j < matrix.length; j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }
    
    public void transposeNoOut(double[][] result, double[][] matrix) {
        double[] temp;
        for (int i = 0; i < matrix[0].length; i++) {
            temp = result[i];

            for (int j = 0; j < matrix.length; j++) {
                temp[j] = matrix[j][i];
            }
        }
    }
}
