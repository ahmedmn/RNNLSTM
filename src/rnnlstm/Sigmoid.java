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
public class Sigmoid {

    public Sigmoid() {
    }

    public double computeSigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double sigmoidOutputToDerivative(double x) {
        return x * (1 - x);
    }

}
