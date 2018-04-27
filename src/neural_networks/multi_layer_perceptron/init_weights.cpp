#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::initWeights(double min, double max) {
  for(int i = 0; i < config.topology.size() - 1; i++) {
    int numRows = config.topology.at(i);
    int numCols = config.topology.at(i + 1);

    MatrixXd w(numRows, numCols);

    for(int rowCounter = 0; rowCounter < numRows; rowCounter++) {
      for(int colCounter = 0; colCounter < numCols; colCounter++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);

        w(rowCounter, colCounter) = dis(gen);
      }
    }

    weights.push_back(w);
  }
};
