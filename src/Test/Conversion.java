package Test;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class Conversion {

//Places each object of data into an array
    public double[][] inputs(String dataFilePath) throws IOException {


        DataInputStream dataInputStream = new DataInputStream((new BufferedInputStream(new FileInputStream(dataFilePath))));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        double[][] inputs = new double[numberOfItems][nRows * nCols];

        for (int i = 0; i < numberOfItems; i++) {
            for (int j = 0; j < nRows * nCols; j++) {
                inputs[i][j] = (double)dataInputStream.readUnsignedByte()/1000.00;
            }
        }

        return inputs;


    }

    public double[][] desired(String labelFilePath) throws IOException {


        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));


        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();
        double[][] desired = new double[numberOfLabels][10];
        for (int i = 0; i < numberOfLabels; i++) {
            desired[i] = this.toTen(labelInputStream.readUnsignedByte());
        }
        return desired;
    }

    public double[] toTen(int n) {
        double[] curr = new double[10];

        for (int i = 0; i < curr.length; i++) {
            if (i == n) {
                curr[i] = 1.0;
            } else {
                curr[i] = 0.0;
            }
        }

        return curr;

    }


}
