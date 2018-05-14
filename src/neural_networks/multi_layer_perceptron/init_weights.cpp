#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::initWeights(double min, double max, int seed) {
  for(int i = 0; i < config.topology.size() - 1; i++) {
    int numRows = config.topology.at(i + 1);
    int numCols = config.topology.at(i);

    MatrixXd w(numRows, numCols);
    MatrixXd dW = MatrixXd::Zero(numRows, numCols);
    MatrixXd bW = MatrixXd::Zero(numRows, numCols);

    for(int rowCounter = 0; rowCounter < numRows; rowCounter++) {
      for(int colCounter = 0; colCounter < numCols; colCounter++) {
        std::random_device rd;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(min, max);

        w(rowCounter, colCounter) = dis(gen);
      }
    }

    weights.push_back(w);
    deltaWeights.push_back(dW);
    bufferWeights.push_back(bW);
  }
};
