#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::feedForward() {
  for(int i = 0; i < (config.topology.size() - 1); i++) {
    layers.at(i + 1)  = (weights.at(i) * layers.at(i)).array() + config.bias;

    if(i == config.topology.size() - 2)
      activate(layers.at(i + 1), config.oActivation);
    else
      activate(layers.at(i + 1), config.hActivation);
  }
};
