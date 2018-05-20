#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::feedForward() {
  for(int i = 0; i < (config.topology.size() - 1); i++) {
    /**
     *  Fetch original rows for this layer excluding bias
     */
    int origRows  = layers.at(i).rows();

    /** 
     *  Resize to include bias neuron
     */
    layers.at(i).conservativeResize(layers.at(i).rows() + 1, layers.at(i).cols());

    /** 
     *  Insert bias neuron
     */
    layers.at(i)(origRows, 0) = config.bias;

    /**
     *  x_(i+1) = wx
     */
    layers.at(i + 1)  = (weights.at(i) * layers.at(i)).array();

    /**
     *  Activate next layer
     */
    if(i == config.topology.size() - 2)
      activate(layers.at(i + 1), config.oActivation);
    else
      activate(layers.at(i + 1), config.hActivation);
  }
};
