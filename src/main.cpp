#include <iostream>
#include <cstdlib>
#include <ctime>
#include <Utility.hpp>
#include <NeuralNetwork.hpp>

int main(void){
    srand(time(0));
    
    int nLayers = 3;
    int LayerNodeCounts[] = {2, 3, 2};
    int inputs[] = {1, 2};
    int expectedOutputs[] = {0, 1};

    NeuralNetwork NN(nLayers, LayerNodeCounts);

    // std::string filePath = "E:/Programming/NeuralNetworks/dataset/mnist_train.csv";
    // trainData data(filePath, 1);

    NN.getOutputs(inputs);

    std::cout << "Before Training: " << NN.Layers[2]->Nodes[0].activiationValue << std::endl;
    std::cout << "Before Training: " << NN.Layers[2]->Nodes[1].activiationValue << std::endl;
    NN.Cost(expectedOutputs);
    std::cout << "Cost Before Training: " << NN.NetworkCost << std::endl << std::endl;

    NN.Train(inputs, expectedOutputs, 200);

    NN.getOutputs(inputs);
    std::cout << "After Training: " << NN.Layers[2]->Nodes[0].activiationValue << std::endl;
    std::cout << "After Training: " << NN.Layers[2]->Nodes[1].activiationValue << std::endl;
    NN.Cost(expectedOutputs);
    std::cout << "Cost After Training: " << NN.NetworkCost << std::endl;
    return 0;
}