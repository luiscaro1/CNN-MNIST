package Layer;

import Neuron.SNeuron;

import java.util.ArrayList;

public class Layer {

    Enum type;
    ArrayList<SNeuron> layerNeurons;

    public Layer(Enum type) {

        this.type = type;
        layerNeurons = new ArrayList<>();
    }

    public Enum getType() {
        return this.type;
    }

    public ArrayList<SNeuron> getLayerNeurons() {
        return layerNeurons;
    }

}
