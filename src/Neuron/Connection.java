package Neuron;

public class Connection {

    SigmoidNeuron receiver;
    double weight;
    double bias;

    public Connection(SigmoidNeuron receiver, double weight, double bias) {


        this.receiver = receiver;
        this.weight = weight;
        this.bias = bias;

    }


    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }


}
