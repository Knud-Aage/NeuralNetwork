package dk.kb.neuralnetworks;

/**
 * Created by Knud Åge on 05-02-2017.
 */
public class Unit {
    double act;           /* Activity of the unit */
    double[] weight;       /* Array of the unit links weights */
    double delta;         /* The units 'delta' (for Back Propagation) */
    double[] prev_dW;      /* Array with last weightchanges of the links
			   (for 'Back Propagation with 'momentum') */
    double[] eW;           /* Array with weights differentiated */
    double[] best_weight;  /* Array with best weight obtained */

}
