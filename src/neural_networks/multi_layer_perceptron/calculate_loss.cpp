#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::calculateLoss(vector<double> target) {
  //int numCols = layers.back().cols();
  int numRows = layers.back().rows();

  loss  = 0.00;

  for(int r = 0; r < numRows; r++) {
    loss += pow((layers.back()(r, 0) - target.at(r)), 2) * 0.5;
    eDerivatives(r, 0) = layers.back()(r, 0) - target.at(r);
  }
}
