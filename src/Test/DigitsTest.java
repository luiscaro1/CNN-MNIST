package Test;

import NeuralNetwork.NeuralNetwork;


import java.io.IOException;


public class DigitsTest {

    public static void main(String[] args) throws IOException, InterruptedException {


        Conversion con = new Conversion();

        double[][] inputs = con.inputs("src/data/train-images.idx3-ubyte");
        double[][] desired = con.desired("src/data/train-labels.idx1-ubyte");

        double[][] trainingInputs = new double[50000][inputs[0].length];
        double[][] trainDesired = new double[50000][desired[0].length];

        System.arraycopy(inputs, 0, trainingInputs, 0, trainingInputs.length);
        System.arraycopy(desired, 0, trainDesired, 0, trainDesired.length);

        NeuralNetwork nn = new NeuralNetwork(new int[]{784, 30, 10});

        //learning rate 0.0314 88-90 accuracy

        nn.train(trainingInputs, trainDesired, 0.0314,10000,10.0);
        double[][] test = new double[10000][inputs[0].length];


        System.arraycopy(inputs, trainingInputs.length, test, 0, test.length);


        nn.predict(test, (image) -> {
            for (int i = 0; i < 784; i += 28) {
                for (int j = 0; j < 28; j++) {

                    if (image[j + i] == 0) {
                        System.out.print("   ");
                    } else {
                        System.out.print(" # ");
                    }


                }

                System.out.println();
            }
        });
    }


}
