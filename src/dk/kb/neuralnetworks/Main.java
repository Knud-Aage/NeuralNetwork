package dk.kb.neuralnetworks;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;


public class Main {

    private static final String FILENAME = "c:\\Code\\Java\\SB\\Innovation\\result.txt";
    private static String[] fna = new String[2];
    private static Dataset train,test,val;
    //BP net = new BP();
    private static long now;
    private static String fname,fname1,fname2,fname3;
    //StringBuffer stringBuffer = new StringBuffer();

    public static void usage() {
        System.out.format("ANTAL SKJULTE KNUDER=%d\n",BP.hidden_units);
        System.out.println("USAGE: BP error <file> <error> <eta> <alfa>\n");
        System.out.println("       BP gen <file> <#epoch> <eta> <alfa>\n");
        System.out.println("       BP test <file>\n");
        System.out.println("       BP auto <file> <#runs> <#epoch> <eta> <alfa>\n");
        System.out.println("       BP analyze <file> <#runs> <#epoch>\n");
    }

    public static void main(String[] args) throws IOException {
        // write your code here


//        Path path = Paths.get("c:\\Code\\Java\\SB\\Innovation\\result.txt");
//        String question = "To be or not to be?";
//        Files.write(path, question.getBytes());

        //FILE *fp,*fopen();

        double trainError = 0.15;
        double eta = 0.1;
        double alfa = 0.1;
        int noEpochs = 500;
        int noRuns = 100;
        int hiddenUnits = 15;

        if (args.equals("error"))
            train(fname1, trainError, eta, alfa, hiddenUnits);
        else if (args.equals("gen"))
            generate(fname1, noEpochs, eta, alfa, hiddenUnits);
        else if (args.equals("test"))
            test(fname1, hiddenUnits);
        else if (args.equals("auto"))
            auto(fname1, noRuns, noEpochs, eta, alfa, hiddenUnits);
        else if (args.equals("analyze"))
            analyze(fname1, noRuns, noEpochs, hiddenUnits);
        else
            usage();
    }

    private static void analyze(String filename, int noRuns, int noEpochs, int hiddenUnits) {
        BP net = new BP();
        net.Set_acc(net, 1.0);
        fna[0] = filename + ".valres";
        fna[1] = filename + ".testres";

        System.out.format("Analyze with %d runs of %d epochs.\n", noRuns, noEpochs);
        train.Read_dataset(filename + ".train", net);
        val.Read_dataset(filename + ".val", net);
        test.Read_dataset(filename + ".test", net);
        net.Allocate_net(net, hiddenUnits);
        net.Initialize_net(net);

        net.Analyze(net, train, val, test, noRuns, noEpochs, filename);
    }

    private static void auto(String filename, int noRuns, int noEpochs, double eta, double alfa, int hiddenUnits) {
        BP net = new BP();
        net.Set_acc(net, 1.0);
        fna[0] = fname + ".valres";
        fna[1] = fname + ".testres";

        System.out.format("Automatic train for %d runs of %d epochs.\n", noRuns,noEpochs);
        train.Read_dataset(fname+".train", net);
        val.Read_dataset(fname1+".val", net);
        test.Read_dataset(fname2+".test", net);
        net.Allocate_net(net, hiddenUnits);
	  /* Initialize_net(&net); */

        net.Set_eta(net, eta);
        net.Set_alfa(net, alfa);

        net.Auto_train(net, train, val, test, noRuns, noEpochs, filename);
    }

    private static void test(String filename, int hiddenUnits) {
        BP net = new BP();
        net.Set_acc(net, 1.0);
        System.out.println("Test the net with with test data.\n");
        test.Read_dataset(filename+".test", net);
        net.Allocate_net(net, hiddenUnits);
        net.Initialize_net(net);

        net.Load_weights(net, filename+".weight");
/*	  Show_weights(&net);*/
        System.out.println("\nNet performance on TEST set\n");
        net.Test_network(net, test);
        net.Print_results(test);
    }

    private static void generate(String filename, int noEpochs, double eta, double alfa, int hiddenUnits) {
        BP net = new BP();
        net.Set_acc(net, 1.0);
        System.out.format("Train for a maximum generalization within %d epochs.\n",noEpochs);
        train.Read_dataset(filename+".train", net);
        val.Read_dataset(filename+".val", net);
        test.Read_dataset(filename+".test", net);
        net.Allocate_net(net, hiddenUnits);
        net.Initialize_net(net);

        net.Set_eta(net, eta);
        net.Set_alfa(net, alfa);

        net.Show_weights(net);
        net.Train_for_gen(net, train, val, filename, noEpochs);
        net.Save_weights(net, filename+".weight");
        System.out.println("\nNet performance on TRAINING set\n");
        net.Test_network(net, train);
        net.Print_results(train);
        System.out.println("\nNet performance on TEST set\n");
        net.Test_network(net, test);
        net.Print_results(test);
    }

    private static void train(String filename, double trainError, double eta, double alfa, int hiddenUnits) {
        BP net = new BP();
        net.Set_acc(net, 1.0);
        System.out.println("Train for error.\n");
        train.Read_dataset(filename+".train", net);
        val.Read_dataset(filename+".val", net);
        test.Read_dataset(filename+".test", net);
        net.Allocate_net(net, hiddenUnits);
        net.Initialize_net(net);

        train.error = test.error = trainError;
        net.Set_eta(net, eta);
        net.Set_alfa(net, alfa);

        net.Show_weights(net);
        net.Train_for_error(net, train, val, fname3);
        net.Save_weights(net,filename+".weight");
        System.out.println("\nNet performance on TRAINING set\n");
        net.Test_network(net, train);
        net.Print_results(train);
        System.out.println("\nNet performance on TEST set\n");
        net.Test_network(net, test);
        net.Print_results(test);
    }
}