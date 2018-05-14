#include "../../../include/neural_networks/MultiLayerPerceptron.hpp"

void MultiLayerPerceptron::saveWeights(string filename) {
  json j = {};

  vector< vector< vector<double> > > weightSet;

  for(int i = 0; i < weights.size(); i++) {
    vector< vector<double> > w;
    for(int j = 0; j < weights.at(i).rows(); j++) {
      vector<double> wv;
      for(int k = 0; k < weights.at(i).cols(); k++) {
        wv.push_back(weights.at(i)(j, k));
      }

      w.push_back(wv);
    }

    weightSet.push_back(w);
  }

  j = weightSet;

  std::ofstream o(filename);
  o << std::setw(4) << j << endl;
}
