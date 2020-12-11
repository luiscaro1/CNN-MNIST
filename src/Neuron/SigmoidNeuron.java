package Neuron;

import java.util.ArrayList;

public abstract class SigmoidNeuron {

    ArrayList<Connection> connections;

    double value;


    public SigmoidNeuron() {

        connections = new ArrayList<>();


    }

    public ArrayList<Connection> connections() {
        return connections;
    }

    public double value() {
        return value;
    }

    public void adjustVal(double newVal) {
        this.value = newVal;
    }
}
