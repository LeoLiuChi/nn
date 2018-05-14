cmake_minimum_required(VERSION 3.7)
project(nn)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++14")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(
  mlp-train
  src/mlp_train.cpp
  src/neural_networks/multi_layer_perceptron/setup.cpp
  src/neural_networks/multi_layer_perceptron/init_weights.cpp
  src/neural_networks/multi_layer_perceptron/init_topology.cpp
  src/neural_networks/multi_layer_perceptron/feed_forward.cpp
  src/neural_networks/multi_layer_perceptron/back_prop.cpp
  src/neural_networks/multi_layer_perceptron/activate.cpp
  src/neural_networks/multi_layer_perceptron/derive.cpp
  src/neural_networks/multi_layer_perceptron/calculate_loss.cpp
  src/neural_networks/multi_layer_perceptron/load_weights.cpp
  src/neural_networks/multi_layer_perceptron/save_weights.cpp
  src/utils/Misc.cpp
)

target_link_libraries(mlp-train Eigen3::Eigen)

add_executable(
  mlp-classify
  src/mlp_classify.cpp
  src/neural_networks/multi_layer_perceptron/setup.cpp
  src/neural_networks/multi_layer_perceptron/init_weights.cpp
  src/neural_networks/multi_layer_perceptron/init_topology.cpp
  src/neural_networks/multi_layer_perceptron/feed_forward.cpp
  src/neural_networks/multi_layer_perceptron/back_prop.cpp
  src/neural_networks/multi_layer_perceptron/activate.cpp
  src/neural_networks/multi_layer_perceptron/derive.cpp
  src/neural_networks/multi_layer_perceptron/calculate_loss.cpp
  src/neural_networks/multi_layer_perceptron/load_weights.cpp
  src/neural_networks/multi_layer_perceptron/save_weights.cpp
  src/utils/Misc.cpp
)

target_link_libraries(mlp-classify Eigen3::Eigen)

add_executable(
  ae-train
  src/ae_train.cpp
  src/neural_networks/multi_layer_perceptron/setup.cpp
  src/neural_networks/multi_layer_perceptron/init_weights.cpp
  src/neural_networks/multi_layer_perceptron/init_topology.cpp
  src/neural_networks/multi_layer_perceptron/feed_forward.cpp
  src/neural_networks/multi_layer_perceptron/back_prop.cpp
  src/neural_networks/multi_layer_perceptron/activate.cpp
  src/neural_networks/multi_layer_perceptron/derive.cpp
  src/neural_networks/multi_layer_perceptron/calculate_loss.cpp
  src/neural_networks/multi_layer_perceptron/load_weights.cpp
  src/neural_networks/multi_layer_perceptron/save_weights.cpp
  src/utils/Misc.cpp
)

target_link_libraries(ae-train Eigen3::Eigen)

add_executable(
  ae-train-images
  src/ae_train_images.cpp
  src/neural_networks/multi_layer_perceptron/setup.cpp
  src/neural_networks/multi_layer_perceptron/init_weights.cpp
  src/neural_networks/multi_layer_perceptron/init_topology.cpp
  src/neural_networks/multi_layer_perceptron/feed_forward.cpp
  src/neural_networks/multi_layer_perceptron/back_prop.cpp
  src/neural_networks/multi_layer_perceptron/activate.cpp
  src/neural_networks/multi_layer_perceptron/derive.cpp
  src/neural_networks/multi_layer_perceptron/calculate_loss.cpp
  src/neural_networks/multi_layer_perceptron/load_weights.cpp
  src/neural_networks/multi_layer_perceptron/save_weights.cpp
  src/neural_networks/autoencoder/vector_to_image.cpp
  src/utils/Misc.cpp
)

target_link_libraries(ae-train-images Eigen3::Eigen)
