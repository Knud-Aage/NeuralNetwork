

//import org.testng.annotations.Test;
// import static org.testng.Assert.*;

import dk.kb.neuralnetworks.Main;
import org.testng.annotations.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


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

   @Test
   public void changeToBlackAndWhite() throws IOException {
       try {
           BufferedImage original = ImageIO.read(new File("Letter_From_Denmark_Page_2.png"));
           BufferedImage binarized = new BufferedImage(original.getWidth(), original.getHeight(), BufferedImage.TYPE_BYTE_BINARY);

           int red;
           int newPixel;
           int threshold = 200;

           for (int i = 0; i < original.getWidth(); i++) {
               for (int j = original.getHeight() / 23; j < 2 * original.getHeight() / 23; j++) {

                   // Get pixels
                   red = new Color(original.getRGB(i, j)).getRed();

                   int alpha = new Color(original.getRGB(i, j)).getAlpha();

                   if (red > threshold) {
                       newPixel = 255;
                   } else {
                       newPixel = 0;
                   }
                   newPixel = colorToRGB(alpha, newPixel, newPixel, newPixel);
                   binarized.setRGB(i, j, newPixel);

               }
           }
           ImageIO.write(binarized, "png", new File("blackwhiteimage.png"));
       } catch (IOException e) {
           e.printStackTrace();
       }
   }

    private static int colorToRGB(int alpha, int red, int green, int blue) {
        int newPixel = 0;
        newPixel += alpha;
        newPixel = newPixel << 8;
        newPixel += red; newPixel = newPixel << 8;
        newPixel += green; newPixel = newPixel << 8;
        newPixel += blue;

        return newPixel;
    }


   //private static final String filename = "C:\\Code\\Repository\\NeuralNetwork\\test\\rug\\rug_450_3000";
   private static final String filename = "/home/kaah/Code/Inno/Cursive OCR/NeuralNetwork/test/rug/rug_450_3000";

}