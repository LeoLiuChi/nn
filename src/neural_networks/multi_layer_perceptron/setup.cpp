#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::setup() {
  initTopology();
  initWeights();

  target        = MatrixXd::Zero(config.topology.back(), 1);
  errors        = MatrixXd::Zero(config.topology.back(), 1);
  eDerivatives  = MatrixXd::Zero(config.topology.back(), 1);
};
