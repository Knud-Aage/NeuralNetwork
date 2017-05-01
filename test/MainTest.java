

//import org.testng.annotations.Test;
// import static org.testng.Assert.*;

import dk.kb.neuralnetworks.Dataset;
import dk.kb.neuralnetworks.Main;

import org.codehaus.groovy.testng.TestNgRunner;
import org.testng.annotations.Test;

/**
 * Created by kaah on 20.02.17.
 */
@Test ()
public class MainTest {
   @Test
   public void testAnalyze() throws Exception {
      int noEpochs = 500;
      int noRuns = 100;
      int hiddenUnits = 15;

      Main.analyze(filename, noRuns, noEpochs, hiddenUnits);
   }

   @Test
   public void testTest1() throws Exception {
      int hiddenUnits = 15;
      Main.test(filename, hiddenUnits);
   }

   @Test
   public void testUsage() throws Exception {
      Main.usage();
   }

   @Test
   public void testAuto() throws Exception {
      double trainError = 90.15;
      double eta = 0.1;
      double alfa = 0.1;
      int noEpochs = 500;
      int noRuns = 100;
      int hiddenUnits = 15;

      Main.auto(filename, noRuns, noEpochs, eta, alfa, hiddenUnits);
      System.out.println("Test");
      Main.usage();

   }

   @Test
   public void testGenerate() throws Exception {
      double trainError = 90.15;
      double eta = 0.1;
      double alfa = 0.1;
      int noEpochs = 500;
      int hiddenUnits = 15;

      Main.generate(filename, noEpochs, eta, alfa, hiddenUnits);
      System.out.println("Test");
      Main.usage();

   }

   @Test
   public void testTrain() throws Exception {
      double trainError = 90.15;
      double eta = 0.1;
      double alfa = 0.1;
      int hiddenUnits = 15;

      Main.train(filename, trainError, eta, alfa, hiddenUnits);
      System.out.println("Test");
      Main.usage();

   }

   private static final String filename = "C:\\Code\\Repository\\NeuralNetwork\\test\\rug\\rug_450_3000";
   private static String[] fna = new String[2];
   private static Dataset train,test,val;


   void testSomething() {



   }

}