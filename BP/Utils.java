package com.neuralnetwork;

/**
 * Created by Knud Åge on 05-02-2017.
 */
public class Utils {

    static public double g(h,a) {
        return (1 / (1 + Math.exp(-(a)*(h))));   /* The activation function */
    }

    static public double wrandom() {
        return ( Math.random()*2 - 1 );
    }
}
