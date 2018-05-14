#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::backProp() {
  for(int i = (config.topology.size() - 1); i > 0; i--) {
    if(i == (config.topology.size() - 1)) {
      gradients[i] = eDerivatives.cwiseProduct(derive(layers[i], config.oActivation));
      deltaWeights[i - 1] = gradients[i] * layers[i - 1].transpose();
    } else {
      gradients[i]  = (weights[i].transpose() * gradients[i + 1]).cwiseProduct(derive(layers[i], config.hActivation));
      deltaWeights[i - 1] = gradients[i] * layers[i - 1].transpose();
    }

  }

  // Update weights
  for(int j = 0; j < weights.size(); j++) {
    weights[j] = (weights[j].array() - (config.learningRate * deltaWeights[j]).array()).matrix();
  }
};
