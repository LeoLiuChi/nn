#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::loadWeights(string filename) {
  std::ifstream i(filename);
  json jWeights;
  i >> jWeights;

  vector< vector< vector<double> > > buffer = jWeights;

  for(int i = 0; i < buffer.size(); i++) {
    for(int j = 0; j < buffer.at(i).size(); j++) {
      for(int k = 0; k < buffer.at(i).at(j).size(); k++) {
        weights[i](j, k) = buffer.at(i).at(j).at(k);
      }
    }
  }
}
