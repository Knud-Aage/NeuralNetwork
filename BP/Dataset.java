package com.neuralnetwork;

/**
 * Created by Knud Åge on 05-02-2017.
 */
public class Dataset {
    int in,out;           /* Integers representing #input, #output */
    int no;               /* No. of patterns in the data set */
    double error;         /* Total error to train the net for */
    Pattern[] Plist;      /* Array of the patterns */
    int no_correct;       /* No.of correct data */
    int no_incorrect;     /* No. of incorrect data */
    double per1,per15,gen;/* The 2 classification percents */

}
