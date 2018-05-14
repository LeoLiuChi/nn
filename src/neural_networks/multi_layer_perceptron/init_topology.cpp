#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::initTopology() {
  for(int i = 0; i < config.topology.size(); i++) {
    printf("Initializing layer %d shape [%d, %d]...\n", i, config.topology.at(i), 1);
    MatrixXd m  = MatrixXd::Zero(config.topology.at(i), 1);
    MatrixXd g  = MatrixXd::Zero(config.topology.at(i), 1);

    layers.push_back(m);
    gradients.push_back(g);
  }
};
