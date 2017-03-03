package dk.kb.neuralnetworks;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Time;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static dk.kb.neuralnetworks.Utils.g;
import static dk.kb.neuralnetworks.Utils.sqr;
import static dk.kb.neuralnetworks.Utils.wrandom;
import static java.lang.Math.sqrt;
import static java.lang.System.exit;

/**
 * Created by Knud Åge on 05-02-2017.
 * /************************************************************************
 * // This program implements a multi-layer perceptron consisting of an
 * // input layer, a number of hidden layers (>=1) and an output layer.
 * //
 * // The net is trained according to the algoritm 'BP'.
 * // The teaching is done incrementally i.e. the weights and
 * // thresholds are adjusted after presentation of each pattern in the
 * // training set.
 * //
 * // A variable 'net' of the type 'Neural_net' has the following records:
 * //
 * // net.no_inputs:                   Integer representing #input
 * // net.no_outputs:                  Integer representing #output
 * // net.no_hidden_layers:            Integer representing #hidden layers
 * // net.no_units[]:                  Array of size of the i'th layer
 * // net.eta, net.alfa:              Learning parameters
 * // net.acc:                         Soft/hardness of activation function
 * // net.input[]:                     Array of the net inputs
 * // net.layer[u].unit[k].act:        Double representing the activity of
 * //                                   the k'th unit in the u'th layer.
 * // net.layer[u].unit[k].weight[j]:  Double representing the weight of the
 * //                                   link between 'unit k in layer u' AND
 * //                                   'unit j in layer u-1'.
 * // net.layer[u].unit[k].delta:      Double representing the units 'delta',
 * //                                   which is computed by back propagation
 * // net.layer[u].unit[k].prev_dW[j]: Double representing the last weight
 * //                                   change of the link between 'unit k in
 * //                                   layer u' AND 'unit j in layer u-1'.
 * //
 * // net.input[0], net.layer[0].act[0], net.layer[1].act[0] e.t.c.
 * // must ALWAYS equal -1.0. (Is done in the initialisation of the net)
 * // The reason for this, is that the weight of a given unit A to the 0'th
 * // unit i the previous layer, simulates the threshold of unit A.
 **************************************************************************/

public class BP {
    int no_inputs;        /* #input units */
    int no_outputs;       /* #output units */
    int no_hidden_layers; /* #hidden layers */
    int[] no_units;        /* Array of the size of each layers in the net */
    double eta, alfa;      /* Learning parameters */
    double acc;           /* Soft/hardness of activation function */
    double[] input;        /* Array of the net inputs */
    Layer[] layer;      /* Array of the nets hidden and output layer */
    static public int hidden_units = 20;

    /* Allocate net with the specifications from the start of the pat.file */
/* Format: #input, #output, #hidden_layer, #hid1_units, #hid2_units ... */
    public BP Allocate_net(BP net, int no_hidden_units) {
        try {
            int prev_size;

            System.out.format("#Hidden units=%d", no_hidden_units);
            net.no_hidden_layers = 1; /* --- Only 1 hidden layer --- */

            net.input = new double[net.no_inputs + 1];
            net.layer = new Layer[net.no_hidden_layers + 1];
            net.no_units = new int[net.no_hidden_layers + 1];
            for (int layer = 0; layer < net.no_hidden_layers; layer++) {
                net.no_units[layer] = no_hidden_units;
            }
            net.no_units[net.no_hidden_layers] = net.no_outputs;
            for (int layer = 0; layer <= net.no_hidden_layers; layer++) {
                net.layer[layer] = new Layer();
                net.layer[layer].unit = new Unit[net.no_units[layer] + 1];
                for (int i = 1; i <= net.no_units[layer]; i++) {
                    net.layer[layer].unit[i] = new Unit();
                    prev_size = ((layer > 0) ? (net.no_units[layer - 1]) : (net.no_inputs));
                    net.layer[layer].unit[i].weight = new double[prev_size + 1];
                    net.layer[layer].unit[i].prev_dW = new double[prev_size + 1];
                    net.layer[layer].unit[i].eW = new double[prev_size + 1];
                    net.layer[layer].unit[i].best_weight = new double[prev_size + 1];
                }
            }
        }
        catch (Exception ex) {
            System.out.print("Exception: " + ex);
        }
        return net;
    }

    /* Initialize weights and threshold to ] -3/sqrt(n); 3/sqrt(n) [     */
/* Also every layers unit 0 is given the activity -1.0               */
    public BP Initialize_net(BP net) {

        try {
            int prev_size;
            net.input[0] = -1.0;
            for (int layerno = 0; layerno <= net.no_hidden_layers; layerno++) {
                net.layer[layerno].unit[0] = new Unit();
                net.layer[layerno].unit[0].act = -1.0;
                prev_size = ((layerno > 0) ? (net.no_units[layerno - 1]) : (net.no_inputs));
                System.out.printf("Layer %d:  Initializing weights to +/- %6.3f\n", layerno, (3 / sqrt(((double) prev_size))));
                for (int i = 1; i <= net.no_units[layerno]; i++) {
                    net.layer[layerno].unit[i] = new Unit();
                    net.layer[layerno].unit[i].weight = new double[prev_size+1];
                    net.layer[layerno].unit[i].prev_dW = new double[prev_size+1];
                    net.layer[layerno].unit[i].eW = new double[prev_size+1];
                    net.layer[layerno].unit[i].best_weight= new double[prev_size+1];
                    for (int j = 0; j <= prev_size; j++) {
                        net.layer[layerno].unit[i].weight[j] = wrandom();
                        net.layer[layerno].unit[i].weight[j] *= (3 / sqrt(((double) prev_size)));
                        net.layer[layerno].unit[i].prev_dW[j] = 0.0;
                        net.layer[layerno].unit[i].eW[j] = 0.0;
                        net.layer[layerno].unit[i].best_weight[j] = 0.0;
                    }
                }
            }
        }
        catch (Exception ex) {
            System.out.print("Exception: " + ex);
        }
        return net;
    }

    /* Set the net parameter 'alfa' (Momentum) */
    void Set_alfa(BP net, double alfa) {
        net.alfa = alfa;
    }

    void Set_eta(BP net, double eta) {
        net.eta = eta;
    }

    /* Set the net parameter 'acc' (Activation function) */
    void Set_acc(BP net, double acc) {
        net.acc = acc;
    }

    /* Show the weights of all the links in the neural net */
    void Show_weights(BP net) {
        int prev_size;

        System.out.printf("\nInput til");
        for (int l = 0; l <= net.no_hidden_layers; l++) {
            System.out.printf("\nLag %d:", l);
            prev_size = ((l > 0) ? (net.no_units[l - 1]) : (net.no_inputs));
            for (int i = 1; i <= net.no_units[l]; i++) {
                System.out.printf("\n");
                for (int j = 0; j <= prev_size; j++) {
                    System.out.printf("%7.3f ", net.layer[l].unit[i].weight[j]);
                }
            }
        }
    }

    /* Save the weights of all the net links */
    void Save_weights(BP net, String filename) {
        try {
            //Get the file reference
            Path path = Paths.get(filename);
            int prev_size;
            String tmpString = "";

//Use try-with-resource to get auto-closeable writer instance
            try (BufferedWriter writer = Files.newBufferedWriter(path)) {
                for (int l = 0; l <= net.no_hidden_layers; l++) {
                    prev_size = ((l > 0) ? (net.no_units[l - 1]) : (net.no_inputs));
                    for (int i = 1; i <= net.no_units[l]; i++) {
                        for (int j = 0; j <= prev_size; j++) {
                            tmpString += String.format("%12.7f ", net.layer[l].unit[i].weight[j]);
                        }
                        writer.write(tmpString);
                    }
                    writer.newLine();
                }
            }
        } catch (IOException ioe) {
            System.out.println("IOE");
        }
    }

    /* Load the weights of the net links */
    void Load_weights(BP net, String filename) {
        List<String> list = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(Paths.get(filename))) {
            list = br.lines().collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
        }


        int prev_size;
        String[] splitStr;

        {
            for (int l = 0; l <= net.no_hidden_layers; l++) {
                prev_size = ((l > 0) ? (net.no_units[l - 1]) : (net.no_inputs));
                for (int i = 1; i <= net.no_units[l]; i++) {
                    splitStr = list.get(0).split(" ");
                    for (int j = 0; j <= prev_size; j++)
                        net.layer[l].unit[i].weight[j] = Double.valueOf(splitStr[j]);
                }
            }
        }
    }

    /* Store the weights of all the net links */
    void Store_weights(BP net) {
        int prev_size;

        for (int layer = 0; layer <= net.no_hidden_layers; layer++) {
            prev_size = ((layer > 0) ? (net.no_units[layer - 1]) : (net.no_inputs));
            for (int i = 1; i <= net.no_units[layer]; i++) {
                for (int j = 0; j <= prev_size; j++) {
                    net.layer[layer].unit[i].best_weight[j] =
                            net.layer[layer].unit[i].weight[j];
                }
            }
        }
    }

    /* Restore the weights of all the net links */
    void Restore_weights(BP net) {
        int prev_size;

        for (int layer = 0; layer <= net.no_hidden_layers; layer++) {
            prev_size = ((layer > 0) ? (net.no_units[layer - 1]) : (net.no_inputs));
            for (int i = 1; i <= net.no_units[layer]; i++) {
                for (int j = 0; j <= prev_size; j++) {
                    net.layer[layer].unit[i].weight[j] =
                            net.layer[layer].unit[i].best_weight[j];
                }
            }
        }
    }

    /* Present the u'te training pattern to the Neural net */
    void Present_pattern(BP net, Dataset data, int u) {
        for (int i = 1; i <= net.no_inputs; i++) {
            net.input[i] = data.Plist[u].input[i];
        }
    }


    /* Compute the activity of all units in the net for the given input */
    void Activate_net(BP net) {
        double sum;

  /* Compute the activity of the 1'st layer using the input layer */
        for (int i = 1; i <= net.no_units[0]; i++) {
            sum = 0.0;
            for (int j = 0; j <= net.no_inputs; j++)
                sum += net.input[j] * net.layer[0].unit[i].weight[j];
            net.layer[0].unit[i].act = g((sum), (net.acc));
      /*printf("(%7.3f %7.3f) ",sum,g((sum),(net.acc)));*/
        }
  /* Compute the activity of the remaining layers */
        for (int l = 1; l <= net.no_hidden_layers; l++) {
            for (int i = 1; i <= net.no_units[l]; i++) {
                sum = 0.0;
                for (int j = 0; j <= net.no_units[l - 1]; j++) {
                    sum += net.layer[l - 1].unit[j].act *
                            net.layer[l].unit[i].weight[j];
                }
                net.layer[l].unit[i].act = g((sum), (net.acc));
      /*printf("(%7.3f,%7.3f) ",sum,net.layer[l].unit[i].act);*/
            }
        }
    }


    /* Compute the summed error of the net on all patterns in the dataset */
    double Error(BP net, Dataset train, Dataset val) {
        double err, maxo;
        int maxi = 1;

  /* Calculate error on TRAINING set + generalization */
        err = 0.0;
        train.per15 = train.per1 = train.gen = 0.0;
        for (int u = 0; u < train.no; u++) {
            Present_pattern(net, train, u);
            Activate_net(net);
            maxo = 0.0;
            for (int i = 1; i <= net.no_outputs; i++) {
                err += sqr((train.Plist[u].output[i]) -
                        (net.layer[net.no_hidden_layers].unit[i].act));
	  /* Find maximum output */
                if (net.layer[net.no_hidden_layers].unit[i].act > maxo) {
                    maxo = net.layer[net.no_hidden_layers].unit[i].act;
                    maxi = i;
                }
            }
            if (train.Plist[u].output[maxi] != 1.0) {
                if (maxi == 1) train.per1 += 1.0;
                else train.per15 += 1.0;
            } else {
                train.gen += 1.0;
            }
        }
        train.per1 /= train.no_incorrect;
        train.per1 *= 100.0;
        train.per15 /= train.no_correct;
        train.per15 *= 100.0;
        train.gen /= train.no;
        train.gen *= 100;

  /* Calculate generalization on VALIDATION set (can be removed) */
        val.per15 = val.per1 = val.gen = 0.0;
        for (int u = 0; u < val.no; u++) {
            Present_pattern(net, val, u);
            Activate_net(net);
            maxo = 0.0;
            for (int i = 1; i <= net.no_outputs; i++) {
	  /* Find maximum output */
                if (net.layer[net.no_hidden_layers].unit[i].act > maxo) {
                    maxo = net.layer[net.no_hidden_layers].unit[i].act;
                    maxi = i;
                }

            }
            if (val.Plist[u].output[maxi] != 1.0) {
                if (maxi == 1) val.per1 += 1.0;
                else val.per15 += 1.0;
            } else {
                val.gen += 1.0;
            }
        }
        val.per1 /= val.no_incorrect;
        val.per1 *= 100.0;
        val.per15 /= val.no_correct;
        val.per15 *= 100.0;
        val.gen /= val.no;
        val.gen *= 100;

        return (0.5 * err);
    }

    /* Compute all units 'delta' by 'error back-propagation'     */
/* It is used that g'(h,a)=a*g(h,a)*(1-g(h,a)) evt. (+0.1)?  */
    void Error_back_propagation(BP net, Dataset train, int u) {
        double sum;

  /* STEP 4 in the BP algorithm p.120 in the book */
        int out_layer = net.no_hidden_layers;
        for (int i = 1; i <= net.no_outputs; i++) {
            net.layer[out_layer].unit[i].delta =
                    (train.Plist[u].output[i] - net.layer[out_layer].unit[i].act) *
                            (net.layer[out_layer].unit[i].act * net.acc *
                                    (1 - net.layer[out_layer].unit[i].act)) + 0.0;
        }
  /* STEP 5 in the BP algorithm p.120 in the book */
        for (int l = net.no_hidden_layers - 1; l >= 0; l--) {
            for (int i = 1; i <= net.no_units[l]; i++) {
                sum = 0.0;
                for (int j = 1; j <= net.no_units[l + 1]; j++) {
                    sum += net.layer[l + 1].unit[j].weight[i] *
                            net.layer[l + 1].unit[j].delta;
                }
                net.layer[l].unit[i].delta = sum * net.acc *
                        (net.layer[l].unit[i].act * (1 - net.layer[l].unit[i].act))
                        + 0.0;
            }
        }

  /* Calculate EW(12) for links between input and first layer */
        for (int i = 0; i <= net.no_inputs; i++) {
            for (int j = 1; j <= net.no_units[0]; j++) {
                net.layer[0].unit[j].eW[i] = net.input[i] *
                        net.layer[0].unit[j].delta;
            }
        }

  /* Calculate EW's for the rest of the layers */
        for (int l = 0; l < net.no_hidden_layers; l++) {
            for (int i = 0; i <= net.no_units[l]; i++) {
                for (int j = 1; j <= net.no_units[l + 1]; j++) {
                    net.layer[l + 1].unit[j].eW[i] =
                            net.layer[l + 1].unit[j].delta *
                                    net.layer[l].unit[i].act;
                }
            }

        }

    }


    /* Update the weights and threshold using the just calculated 'delta' */
/* This procedure performs STEP 6 in the BP algorithm */
    void Update_weights(BP net) {
        double dW;

  /* Beregn vaegtforandringerne for de "sidste" lag i nettet */
        for (int l = net.no_hidden_layers; l >= 1; l--) {
            for (int i = 1; i <= net.no_units[l]; i++) {
                for (int j = 0; j <= net.no_units[l - 1]; j++) {
                    dW = net.eta * net.layer[l].unit[i].eW[j];
                    dW += net.alfa * net.layer[l].unit[i].prev_dW[j];
                    net.layer[l].unit[i].weight[j] += dW;
                    net.layer[l].unit[i].prev_dW[j] = dW;
                }
            }
        }
  /* Beregn vaegtforandringen for forbindelsen mellem input laget og */
  /* det efterfoelgende lag.                                         */
        for (int i = 1; i <= net.no_units[0]; i++) {
            for (int j = 0; j <= net.no_inputs; j++) {
                dW = net.eta * net.layer[0].unit[i].eW[j];
                dW += net.alfa * net.layer[0].unit[i].prev_dW[j];
                net.layer[0].unit[i].weight[j] += dW;
                net.layer[0].unit[i].prev_dW[j] = dW;
            }
        }
    }

    /* Evaluate the net on a test set (calculate ability of generalization) */
    void Test_network(BP net, Dataset data) {
        int output = 1;
        double maxo;
        long tid1, tid2;

        data.per1 = data.per15 = data.gen = 0.0;
        for (int u = 0; u < data.no; u++) {
            Present_pattern(net, data, u);
            Activate_net(net);
            maxo = 0.0;
            for (int i = 1; i <= net.no_outputs; i++) {
                if (net.layer[net.no_hidden_layers].unit[i].act > maxo) {
                    maxo = net.layer[net.no_hidden_layers].unit[i].act;
                    output = i;
                }
            }
            if (data.Plist[u].output[output] != 1.0) {
                if (output == 1) data.per1 += 1.0;
                else data.per15 += 1.0;
            } else {
                data.gen++;
            }
        }

        data.per1 /= data.no_incorrect;
        data.per1 *= 100.0;
        data.per15 /= data.no_correct;
        data.per15 *= 100.0;
        data.gen /= data.no;
        data.gen *= 100;
        //printf("Time: %ld   Performance: %d\n",tid2-tid1,data.no/(tid2-tid1));
    }

    void Print_results(Dataset data) {
        System.out.println("Correctly predicted in class / total in class\n");
        System.out.println("---------------------------------------------\n");
        System.out.format("Incorrect seed classified incorrectly (<1%%):  (#incorrect=%d) %7.3f %%\n", data.no_incorrect, data.per1);
        System.out.format("Correct seed classified incorrectly (<15%%):   (#correct=%d) %7.3f %%\n", data.no_correct, data.per15);
        System.out.format("Generalization:  %7.3f\n", data.gen);
    }

    /* Train the net until the error is less than the given value */
/* The error is written as a function of #epoch in a file */
    void Train_for_error(BP net, Dataset train, Dataset val, String fname) {
        Permute_set set = new Permute_set();
        int u, epoch;
        double err;
        //Get the file reference
        int prev_size;

//Use try-with-resource to get auto-closeable writer instance
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fname + ".error"))) {
            writer.write(String.format("# eta=%7.3f, alfa=%7.3f, error=%7.2f\n", net.eta, net.alfa, train.error));
        } catch (IOException ioe) {
            System.out.println("IOE");
        }
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fname + ".val1"))) {
            writer.write(String.format("# eta=%7.3f, alfa=%7.3f, error=%7.2f", net.eta, net.alfa, train.error));
        } catch (IOException ioe) {
            System.out.println("IOE");
        }
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fname + ".val15"))) {
            writer.write(String.format("# eta=%7.3f, alfa=%7.3f, error=%7.2f\n", net.eta, net.alfa, train.error));
        } catch (IOException ioe) {
            System.out.println("IOE");
        }


        //Allocate_set(&set,train.no);
        err = Error(net, train, val);
        System.out.format("Start error=%7.3f\n", err);
        epoch = 0;
        Permute_set trainSet = new Permute_set();
        while (err > train.error) {
            set.Init_set(set, train.no);
            epoch++;
            System.out.format("Epoch %d.  ", epoch);
            while (set.Size_of_set(set) != 0) {
                u = set.Get_random_element(set);
                Present_pattern(net, train, u);
                Activate_net(net);
                Error_back_propagation(net, train, u);
                Update_weights(net);
            }

            err = Error(net, train, val);
            System.out.format("Error=%7.3f\n", err);
            try (FileWriter f = new FileWriter(fname + ".error", true);
                 BufferedWriter b = new BufferedWriter(f);
                 PrintWriter p = new PrintWriter(b);) {
                p.println(String.format("%d %7.3f\n", epoch, err));
            } catch (IOException i) {
                i.printStackTrace();
            }

            try (FileWriter f = new FileWriter(fname + ".error", true);
                 BufferedWriter b = new BufferedWriter(f);
                 PrintWriter p = new PrintWriter(b);) {
                p.println(String.format("%d %7.3f\n", epoch, val.per1));
            } catch (IOException i) {
                i.printStackTrace();
            }
            try (FileWriter f = new FileWriter(fname + ".error", true);
                 BufferedWriter b = new BufferedWriter(f);
                 PrintWriter p = new PrintWriter(b);) {
                p.println(String.format("%d %7.3f\n", epoch, val.per15));
            } catch (IOException i) {
                i.printStackTrace();
            }
        }
    }

    /* Train the net for a fixed #epoch, and return the best net */
    void Train_for_gen(BP net, Dataset train, Dataset val, String fname, int no_epoch) {
        Permute_set set = new Permute_set();
        int u, epoch, minepoch;
        double err, gen, mingen, gen1, gen15;
        String[] fna = new String[2];
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fname + ".error"))) {
            writer.write(String.format("# eta=%7.3f, alfa=%7.3f, #epoch=%d\n", net.eta, net.alfa, no_epoch));
        } catch (IOException ioe) {
            System.out.println("IOE");
        }
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fname + ".val1"))) {
            writer.write(String.format("# eta=%7.3f, alfa=%7.3f, #epoch=%d\n", net.eta, net.alfa, no_epoch));
        } catch (IOException ioe) {
            System.out.println("IOE");
        }
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fname + ".val15"))) {
            writer.write(String.format("# eta=%7.3f, alfa=%7.3f, #epoch=%d\n", net.eta, net.alfa, no_epoch));
        } catch (IOException ioe) {
            System.out.println("IOE");
        }

        err = Error(net, train, val);
        System.out.format("Start error=%7.3f\n", err);
        minepoch = epoch = 0;
        mingen = 999.0;
        gen1 = gen15 = 0.0;
        while (epoch < no_epoch) {
            set.Init_set(set, train.no);
            epoch++; /* printf("Epoch %d.  ",epoch); */
            while (set.Size_of_set(set) != 0) {
                u = set.Get_random_element(set);
                Present_pattern(net, train, u);
                Activate_net(net);
                Error_back_propagation(net, train, u);
                Update_weights(net);
            }

            err = Error(net, train, val);
            gen = Utils.abs((val.per15 >= 15.0) ? 1.0 : 0.0) * 100.0 + val.per1;
            if ((gen < mingen) && (gen < 100.0)) {
                mingen = gen;
                gen1 = val.per1;
                gen15 = val.per15;
                minepoch = epoch;
                Store_weights(net);
            }

      /* printf("Error=%7.3f\n",err); */
            try (FileWriter f = new FileWriter(fname + ".error", true);
                 BufferedWriter b = new BufferedWriter(f);
                 PrintWriter p = new PrintWriter(b);) {
                p.println(String.format("%d %7.3f\n", epoch, err));
            } catch (IOException i) {
                i.printStackTrace();
            }

            try (FileWriter f = new FileWriter(fname + ".val1", true);
                 BufferedWriter b = new BufferedWriter(f);
                 PrintWriter p = new PrintWriter(b);) {
                p.println(String.format("%d %7.3f\n", epoch, val.per1));
            } catch (IOException i) {
                i.printStackTrace();
            }
            try (FileWriter f = new FileWriter(fname + ".val15", true);
                 BufferedWriter b = new BufferedWriter(f);
                 PrintWriter p = new PrintWriter(b);) {
                p.println(String.format("%d %7.3f\n", epoch, val.per15));
            } catch (IOException i) {
                i.printStackTrace();
            }
        }
        System.out.format("Maximum performance reached after %d epochs.\n", minepoch);
        System.out.format("Incorrect seed classified incorrectly (<1%%): %7.3f%%\n", gen1);
        System.out.format("Correct seed classified incorrectly (<15%%): %7.3f%%\n\n", gen15);

        Restore_weights(net);
    }

    void Auto_train(BP net, Dataset train, Dataset val, Dataset test, int no_runs, int no_epoch, String fname) {
        Permute_set set = new Permute_set();
        int u, minepoch, epoch;
        long Stid, Sperf, SSQtid, SSQperf, Sepoch, SSQepoch;
        long tid1, tid2, tid3, tid4;
        double err, gen, mingen, gen1, gen15;
        double VSgen1, VSSQgen1, VSgen15, VSSQgen15, Sgen1, SSQgen1, Sgen15, SSQgen15;
        String[] fna = new String[2];
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fname + ".valres"))) {
            writer.write(String.format("# eta=%7.3f, alfa=%7.3f, epoch=%d, hidden=%d\n",
                    net.eta, net.alfa, no_epoch, net.no_units[0]));
        } catch (IOException ioe) {
            System.out.println("IOE");
        }
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fname + ".testres"))) {
            writer.write(String.format("# eta=%7.3f, alfa=%7.3f, epoch=%d, hidden=%d\n",
                    net.eta, net.alfa, no_epoch, net.no_units[0]));
        } catch (IOException ioe) {
            System.out.println("IOE");
        }

        minepoch = 0;
        gen1 = gen15 = 0.0;

        Sepoch = Stid = Sperf = SSQepoch = SSQtid = SSQperf = 0;
        Sgen1 = SSQgen1 = Sgen15 = SSQgen15 = 0.0;
        VSgen1 = VSSQgen1 = VSgen15 = VSSQgen15 = 0.0;
        for (int run = 0; run < no_runs; run++) {
            tid1 = System.currentTimeMillis();
            Initialize_net(net);
            mingen = 999.0;
            for (epoch = 0; epoch < no_epoch; epoch++) {
                set.Init_set(set, train.no);
	  /*printf("Run: %d   Epoch: %d.\n",run,epoch);*/
                while (set.Size_of_set(set) != 0) {
                    u = set.Get_random_element(set);
                    Present_pattern(net, train, u);
                    Activate_net(net);
                    Error_back_propagation(net, train, u);
                    Update_weights(net);
                }

                err = Error(net, train, val);
                gen = Utils.abs((val.per15 > 15.0) ? 1.0 : 0.0) * 100.0 + val.per1;
                if ((gen < mingen)) {
                    mingen = gen;
                    gen1 = val.per1;
                    gen15 = val.per15;
                    minepoch = epoch;
                    Store_weights(net);
                    System.out.format("Epoch: %d   storing 1 pct %.2f 15 pct %.2f \n", epoch, val.per1, val.per15);
                }

            }
            tid2 = System.currentTimeMillis();
            Restore_weights(net);
            tid3 = System.currentTimeMillis();
            Test_network(net, test);
            tid4 = System.currentTimeMillis();

            Sepoch += minepoch;
            SSQepoch += minepoch * minepoch;
            VSgen1 += gen1;
            VSSQgen1 += gen1 * gen1;
            VSgen15 += gen15;
            VSSQgen15 += gen15 * gen15;
            Sgen1 += test.per1;
            SSQgen1 += test.per1 * test.per1;
            Sgen15 += test.per15;
            SSQgen15 += test.per15 * test.per15;
            Stid = Stid + (tid2 - tid1);
            SSQtid = SSQtid + (tid2 - tid1) * (tid2 - tid1);
            Sperf = Sperf + (test.no / (tid4 - tid3));
            SSQperf = SSQperf + (test.no / (tid4 - tid3)) * (test.no / (tid4 - tid3));
            try (FileWriter f = new FileWriter(fname + ".valres", true);
                 BufferedWriter b = new BufferedWriter(f);
                 PrintWriter p = new PrintWriter(b);) {
                p.println(String.format("%d %8.4f %8.4f\n", minepoch, gen1, gen15));
            } catch (IOException i) {
                i.printStackTrace();
            }
            try (FileWriter f = new FileWriter(fname + ".testres", true);
                 BufferedWriter b = new BufferedWriter(f);
                 PrintWriter p = new PrintWriter(b);) {
                p.println(String.format("%d %8.4f %8.4f %d %d %d\n",
                        minepoch, test.per1, test.per15, tid2 - tid1, test.no / (tid4 - tid3), (tid4 - tid3)));
            } catch (IOException i) {
                i.printStackTrace();
            }

        }
        Sepoch /= no_runs;
        SSQepoch /= no_runs;
        VSgen1 /= no_runs;
        VSgen15 /= no_runs;
        VSSQgen1 /= no_runs;
        VSSQgen15 /= no_runs;
        Sgen1 /= no_runs;
        Sgen15 /= no_runs;
        SSQgen1 /= no_runs;
        SSQgen15 /= no_runs;
        Stid /= no_runs;
        SSQtid /= no_runs;
        Sperf /= no_runs;
        SSQperf /= no_runs;
        try (FileWriter f = new FileWriter(fname + ".valres", true);
             BufferedWriter b = new BufferedWriter(f);
             PrintWriter p = new PrintWriter(b);) {
            p.println(String.format("Epoch: (%d %7.3f)    1%%: (%7.3f %7.3f)   15%%: (%7.3f %7.3f)\n",
                    Sepoch, sqrt(((double) (SSQepoch - Sepoch * Sepoch))),
                    VSgen1, sqrt((VSSQgen1 - VSgen1 * VSgen1)),
                    VSgen15, sqrt((VSSQgen15 - VSgen15 * VSgen15))));
        } catch (IOException i) {
            i.printStackTrace();
        }
        try (FileWriter f = new FileWriter(fname + ".testres", true);
             BufferedWriter b = new BufferedWriter(f);
             PrintWriter p = new PrintWriter(b);) {
            p.println(String.format("Epoch: (%d %7.3f)    1%%: (%7.3f %7.3f)   15%%: (%7.3f %7.3f)\n",
                    Sepoch, sqrt(((double) (SSQepoch - Sepoch * Sepoch))),
                    Sgen1, sqrt((SSQgen1 - Sgen1 * Sgen1)),
                    Sgen15, sqrt((SSQgen15 - Sgen15 * Sgen15))));
            p.println(String.format("Time: (%d %.2f)    Performance: (%d %.2f)\n",
                    Stid, sqrt(((double) (SSQtid - Stid * Stid))),
                    Sperf, sqrt(((double) (SSQperf - Sperf * Sperf)))));
        } catch (IOException i) {
            i.printStackTrace();
        }
    }

    /* Analyze with use of different parameter values */
    void Analyze(BP net, Dataset train, Dataset val, Dataset test, int no_runs, int no_epoch, String fname) {
        for (double eta = 0.1; eta <= 0.85; eta += 0.15) {
            for (double alfa = 0.1; alfa <= 1.0; alfa += 0.20) {
                Set_eta(net, eta);
                Set_alfa(net, alfa);
                Auto_train(net, train, val, test, no_runs, no_epoch, fname);
            }
        }
    }
}