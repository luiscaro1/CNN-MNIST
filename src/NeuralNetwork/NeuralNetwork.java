package NeuralNetwork;

import Layer.Layer;
import Layer.LayerType;
import Neuron.Connection;
import Neuron.SNeuron;
import Util.Draw;
import Util.Matrix;

import java.util.ArrayList;

import java.util.concurrent.TimeUnit;

public class NeuralNetwork {

    private Object[] layers;


    //net determines the structure of the network, each index represents a layer and the amount of Neurons in that layer.
    public NeuralNetwork(int[] net) {


        //We initialize the buckets where the layers are going to reside.
        layers = new Object[net.length];


        // We go through every bucket and add a layer
        for (int i = 0; i < layers.length; i++) {

            //An enum is going to determine the layer type
            LayerType currentType;

            if (i == 0) {
                currentType = LayerType.INPUT;
            } else if (i == layers.length - 1) {
                currentType = LayerType.OUTPUT;
            } else {
                currentType = LayerType.HIDDEN;
            }


            layers[i] = new Layer(currentType);
            Layer layer = (Layer) layers[i];

            //We add new neurons to the layer (net[i])

            for (int j = 0; j < net[i]; j++) {

                layer.getLayerNeurons().add(new SNeuron());

            }


        }


        this.connect();

    }


    //Establishes the proper connections between neurons of adjacent layers layers. (One sided connection)
    private void connect() {


        for (int i = 0; i < this.layers.length - 1; i++) {

            //neuron from the current layer(sender)
            Layer currentLayer = (Layer) this.layers[i];
            //neuron from the next layer (receiver)
            Layer nextLayer = (Layer) this.layers[i + 1];
            SNeuron currentNeuron;

            for (int j = 0; j < currentLayer.getLayerNeurons().size(); j++) {
                //we establish a connection for each current layer neuron with all of the neurons in the next layer
                currentNeuron = currentLayer.getLayerNeurons().get(j);
                SNeuron ntl;

                for (int k = 0; k < nextLayer.getLayerNeurons().size(); k++) {

                    //Node to link to
                    ntl = nextLayer.getLayerNeurons().get(k);
                    //amount of neurons in the layer


                    //simplified Xavier initialization
                    double randomWeight = Math.random() * 2.0*(1 / Math.sqrt(nextLayer.getLayerNeurons().size())) -(1 / Math.sqrt(nextLayer.getLayerNeurons().size()));


                    //Making the connection with the random weight and the receiving neuron. Good practice to make initial biases 0.
                    currentNeuron.connections().add(new Connection(ntl, randomWeight, 0.0));


                }


            }


        }


    }

    //Converts a layer's weights into a matrix

    private double[][] weightsToMatrix(Layer layer) {

        ArrayList<SNeuron> neurons = layer.getLayerNeurons();

        double[][] result = new double[neurons.get(0).connections().size()][neurons.size()];

        for (int i = 0; i < neurons.size(); i++) {

            SNeuron currentNeuron = neurons.get(i);

            for (int j = 0; j < neurons.get(i).connections().size(); j++) {

                result[j][i] = currentNeuron.connections().get(j).getWeight();

            }
        }


        return result;


    }

    //Converts a layer's biases into a matrix
    private double[][] biasesToMatrix(Layer layer) {

        ArrayList<SNeuron> neurons = layer.getLayerNeurons();

        double[][] result = new double[neurons.get(0).connections().size()][neurons.size()];

        for (int i = 0; i < neurons.size(); i++) {

            SNeuron currentNeuron = neurons.get(i);

            for (int j = 0; j < neurons.get(i).connections().size(); j++) {

                result[j][i] = currentNeuron.connections().get(j).getBias();

            }
        }


        return result;


    }

    //Converts the values/ outputs of a Layer into a matrix
    private double[][] valuesToMatrix(Layer layer) {

        ArrayList<SNeuron> neurons = layer.getLayerNeurons();

        double[][] values = new double[1][neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            values[0][i] = neurons.get(i).value();
        }

        return values;


    }

    //Sets all the neuron values in the layer to the values in the newVal matrix
    private void adjustValues(Layer layer, double[][] newVal) {

        ArrayList<SNeuron> adj = layer.getLayerNeurons();

        for (int i = 0; i < adj.size(); i++) {
            adj.get(i).adjustVal(newVal[i][0]);
        }

    }

    //feeds the inputs to the neural network from the input layer all the way to the output layer.
    private void feedForward(double[] inputs) {

        this.addInputs(inputs);
        //we start in the first hidden layer
        for (int i = 0; i < this.layers.length - 1; i++) {
            //has the values and the weights
            Layer currentLayer = (Layer) layers[i];
            //has the receiving neurons
            Layer adjustLayer = (Layer) layers[i + 1];

            //the weights between sender and receiver neurons in matrix form
            double[][] currentWeights = this.weightsToMatrix(currentLayer);

            double[][] currentBiases = this.biasesToMatrix(currentLayer);
            //values  of the receiver neurons in matrix form
            double[][] currentValues = this.valuesToMatrix(currentLayer);
            // dot product of weights and values.
            double[][] sum = Matrix.add(Matrix.multiply(currentWeights, Matrix.transpose(currentValues)), currentBiases);
            //adjust receiving layer values with sum and activation function
            this.adjustValues(adjustLayer, this.activate(sum));


        }


    }

    //Adds the input values to the input nodes
    private void addInputs(double[] inputs) {


        Layer inputLayer = (Layer) layers[0];

        ArrayList<SNeuron> n = inputLayer.getLayerNeurons();

        if (n.size() != inputs.length)
            throw new IllegalArgumentException("The inputs dimensions should match the input neuron dimensions!");

        for (int i = 0; i < n.size(); i++)
            n.get(i).adjustVal(inputs[i]);
    }


    //predicts what the data input is in terms of classifications learned from training.

    public void predict(double[][] inputs) throws InterruptedException {
        this.predict(inputs, null);
    }


    public void predict(double[][] inputs, Draw d) throws InterruptedException {
        TimeUnit.SECONDS.sleep(5);
        for (double[] input : inputs) {
            this.feedForward(input);
            //prints the data to the terminal with the desired format.
            if (d != null) {
                d.show(input);
            }


            Layer output = (Layer) this.layers[this.layers.length - 1];
            ArrayList<SNeuron> neurons = output.getLayerNeurons();

            double max = neurons.get(0).value();
            int index = 0;
            for (int j = 1; j < neurons.size(); j++) {
                SNeuron curr = neurons.get(j);
                if (max < curr.value()) {
                    index = j;
                    max = curr.value();
                }
            }

            System.out.println("Prediction: " + index);

            TimeUnit.SECONDS.sleep(5);


        }
    }


    //trains the neural network with data

    public void train(double[][] inputs, double[][] desired, double learningRate, int batch, double error) {
        this.train(inputs,desired,learningRate,batch,0,error);
    }

    public void train(double[][] inputs, double[][] desired, double learningRate, int batch, int epochs) {
        this.train(inputs,desired,learningRate,batch,epochs,0);
    }


    private void train(double[][] inputs, double[][] desired, double learningRate, int batch, int epochs, double error) {


        int ep;

        if (epochs == 0) {
            ep = 1;
        } else {
            ep = epochs;
        }


        for (int i = 0; i < ep; i++) {
            int right = 0;
            double[][] currentInput = new double[batch][inputs[0].length];
            double[][] currentDesired = new double[batch][desired[0].length];


            int n = (int) (Math.random() * ((inputs.length / batch) - 1));

            System.arraycopy(inputs, n * batch, currentInput, 0, batch);
            System.arraycopy(desired, n * batch, currentDesired, 0, batch);


            for (int j = 0; j < currentInput.length; j++) {


                //We feed the neural network and it feeds forward.
                this.feedForward(currentInput[j]);


                ////////////////////////////////////////////////
                Layer OPLayer = (Layer) this.layers[this.layers.length - 1];

                ArrayList<SNeuron> neurons = OPLayer.getLayerNeurons();
                int index = 0;
                double max = neurons.get(0).value();
                int actual = 0;


                for (int l = 0; l < neurons.size(); l++) {

                    if (currentDesired[j][l] == 1) actual = l;
                    if (neurons.get(l).value() > max) {
                        max = neurons.get(l).value();
                        index = l;
                    }

                }

                if (index == actual) {
                    right++;
                }


                //Now we need to back propagate to readjust the weights

                this.backProp(currentDesired[j], learningRate);
            }
            if ((100 - (100.0 * right / currentInput.length) > error) && epochs == 0) {
                ep++;
            }

            System.out.print("Epoch " + (i + 1) + " | " + "score: " + right + "/" + currentInput.length + " percent: ");
            System.out.format("%.2f", (100.0 * right / currentInput.length));
            System.out.println("%");

        }

    }

    //adjusts each layer's weights and biases (Learning)
    private void backProp(double[] desired, double learningRate) {

        //the current error
        double[][] wError = null;
        double[][] bError = null;


        //we start from the neuron second to last, and adjust all the weights until we reach the input layer, hence back-propagation.
        for (int i = layers.length - 2; i >= 0; i--) {
            //current layer has the weights for the layer we want to adjust, next layer has the values we want to adjust
            Layer currentLayer = (Layer) this.layers[i];
            Layer nextLayer = (Layer) this.layers[i + 1];


            // weight matrix of adjusting nextLayer
            double[][] CWM = this.weightsToMatrix(currentLayer);
            double[][] CBM = this.biasesToMatrix(currentLayer);

            if (nextLayer.getType().equals(LayerType.OUTPUT)) {

                //Error in the output layer is desired - output
                wError = Matrix.subtract(Matrix.transpose(desired), Matrix.transpose(this.valuesToMatrix(nextLayer)));
                bError = wError;

            } else {

                double[][] NWM = this.weightsToMatrix(nextLayer);
                double[][] NBM = this.biasesToMatrix(nextLayer);

                //Error in hidden layers Weights^T * (error from last layer)
                assert wError != null;
                if (wError.length == 1 && wError[0].length == 1) {

                    wError = Matrix.constant(Matrix.transpose(NWM), wError[0][0]);
                    bError = Matrix.constant(Matrix.transpose(NBM), bError[0][0]);

                } else {
                    wError = Matrix.multiply(Matrix.transpose(NWM), wError);
                    bError = Matrix.multiply(Matrix.transpose(NWM), bError);
                }


            }


            // have to make a n x 1 matrix to subtract to the sigmoid;
            double[][] one = new double[nextLayer.getLayerNeurons().size()][1];


            for (int j = 0; j < nextLayer.getLayerNeurons().size(); j++) {
                one[j][0] = 1;
            }
            double[][] CVM = this.valuesToMatrix(currentLayer);

            //sum of all weights (currentLayer) with values(nextLayer)
            double[][] sum = Matrix.transpose(this.valuesToMatrix(nextLayer));


            double[][] wFirstTerm = Matrix.constant(Matrix.oneToOne(Matrix.oneToOne(wError, this.activate(sum)), Matrix.subtract(one, activate(sum))), learningRate);
            double[][] bFirstTerm = Matrix.constant(Matrix.oneToOne(Matrix.oneToOne(bError, this.activate(sum)), Matrix.subtract(one, activate(sum))), learningRate);

            //Gradient descent
            double[][] wgd = Matrix.multiply(wFirstTerm, CVM);
            double[][] bgd = Matrix.multiply(bFirstTerm, CVM);

            double[][] NW = Matrix.add(CWM, wgd);
            double[][] NB = Matrix.add(CBM, bgd);


            //adjust weights

            this.adjustWeights(currentLayer, NW);
            this.adjustBiases(currentLayer, NB);


        }

    }

    //replaces all the weights in the current layer with the values of the nw (new weight) matrix
    private void adjustWeights(Layer currentLayer, double[][] nw) {


        ArrayList<SNeuron> neurons = currentLayer.getLayerNeurons();


        for (int i = 0; i < neurons.size(); i++) {
            for (int j = 0; j < neurons.get(i).connections().size(); j++) {

                neurons.get(i).connections().get(j).setWeight(nw[j][i]);
            }
        }
    }

    //replaces all the biases in the current layer with the values of the nw (new biases) matrix

    private void adjustBiases(Layer currentLayer, double[][] nb) {


        ArrayList<SNeuron> neurons = currentLayer.getLayerNeurons();


        for (int i = 0; i < neurons.size(); i++) {
            for (int j = 0; j < neurons.get(i).connections().size(); j++) {

                neurons.get(i).connections().get(j).setBias(nb[j][i]);
            }
        }
    }


    //for the feeding forward
    private double activate(double z) {


        return 1.0 / (1.0 + Math.pow(Math.E, (-1 * z)));


    }


    //for the gradient descent
    private double[][] activate(double[][] M) {

        double[][] result = new double[M.length][M[0].length];
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[i].length; j++) {

                result[i][j] = this.activate(M[i][j]);


            }
        }


        return result;

    }


}
