package dk.kb.neuralnetworks;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import java.util.StringJoiner;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by Knud Åge on 05-02-2017.
 */
public class Dataset {
    int in,out;           /* Integers representing #input, #output */
    int no;               /* No. of patterns in the data set */
    double error;         /* Total error to train the net for */
    int no_correct;       /* No.of correct data */
    int no_incorrect;     /* No. of incorrect data */
    Pattern[] Plist;      /* Array of the patterns */
    double per1,per15,gen;/* The 2 classification percents */

    /* Read the training set from file */
    public void Read_dataset(String filename, BP net)
    {
        int temp;
        List<String> list = new ArrayList<>();
/*
        try (BufferedReader br = Files.newBufferedReader(Paths.get(filename))) {
            list = br.lines().collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
        }
*/

        in = Integer.valueOf(list.get(0));
        list.remove(0);
        out = Integer.valueOf(list.get(0));
        list.remove(0);
        no = Integer.valueOf(list.get(0));
        list.remove(0);
        Plist = new Pattern[no];
        String[] splitStr;
        for (int u=0; u < no; u++)
        {
            Plist[u] = new Pattern();
            Plist[u].input= new double[in+1];
            Plist[u].output= new double[out+1];
            splitStr = list.get(0).split(" ");
            for (int j=0; j < splitStr.length-2; j++)
                Plist[u].input[j] = Double.valueOf(splitStr[j]);
            temp = Integer.valueOf(splitStr[splitStr.length-1]);
            Plist[u].patternClass=temp;
            if (temp==0)
                no_correct++;
            else
                no_incorrect++;
            for (int i=1; i <= out; i++)
            {
                Plist[u].output[i]=((i==(temp+1)) ? (1.0) : (0.0));
            }
            list.remove(0);
        }
        /*list.remove(0);
        splitStr = list.get(0).split(" ");
        net.no_inputs = Integer.valueOf(splitStr[splitStr.length-1]);
        list.remove(0);
        splitStr = list.get(0).split(" ");
        net.no_outputs = Integer.valueOf(splitStr[splitStr.length-1]);
        no = Integer.valueOf(list.get(splitStr.length-1));
        */
        net.no_inputs = in;
        net.no_outputs = out;
        System.out.format("#Data: %d",no);
        System.out.format("  Inputs: %d",net.no_inputs);
        System.out.format("  Outputs: %d\n",net.no_outputs);
        in = net.no_inputs;
        out = net.no_outputs;
        //Plist= new Pattern[no];
        //no_correct=0; no_incorrect=0;
    }
}
