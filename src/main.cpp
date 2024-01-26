#include <iostream>
#include <cstdlib>
#include <ctime>
#include <Utility.hpp>
#include <NeuralNetwork.hpp>

int main(void){
    srand(time(0));
    
    int nLayers = 3;
    int LayerNodeCounts[] = {784, 70, 10};

    NeuralNetwork NN(nLayers, LayerNodeCounts);

    std::string trainPath = "E:/Programming/NeuralNetworks/dataset/mnist_train.csv";
    std::string testPath = "E:/Programming/NeuralNetworks/dataset/mnist_test.csv";
    trainData data(trainPath, 2000);
    trainData test(testPath);

    NN.Train(&data, 50, 200, 500);

    std::cout << "\n" << NN.getAccuracy(&test);
    return 0;
}