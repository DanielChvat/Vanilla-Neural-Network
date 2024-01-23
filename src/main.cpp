#include <iostream>
#include <cstdlib>
#include <ctime>
#include <Utility.hpp>
#include <NeuralNetwork.hpp>

int main(void){
    srand(time(0));
    
    int nLayers = 3;
    int LayerNodeCounts[] = {10, 30, 5};
    double inputs[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    int expectedOutputs[] = {-1, -1, -1, 1, -1};

    NeuralNetwork NN(nLayers, LayerNodeCounts);

    // std::string filePath = "E:/Programming/NeuralNetworks/dataset/mnist_train.csv";
    // trainData data(filePath, 1000);

    NN.getOutputs(inputs);

    std::cout << "Before Training: " << NN.Layers[2]->Nodes[0].activiationValue << std::endl;
    std::cout << "Before Training: " << NN.Layers[2]->Nodes[1].activiationValue << std::endl;
    std::cout << "Before Training: " << NN.Layers[2]->Nodes[2].activiationValue << std::endl;
    std::cout << "Before Training: " << NN.Layers[2]->Nodes[3].activiationValue << std::endl;
    std::cout << "Before Training: " << NN.Layers[2]->Nodes[4].activiationValue << std::endl;

    NN.Cost(expectedOutputs);
    std::cout << "Cost Before Training: " << NN.NetworkCost << std::endl << std::endl;

    NN.Train(inputs, expectedOutputs, 10, 15);

    std::cout << std::endl;
    NN.getOutputs(inputs);
    std::cout << "After Training: " << NN.Layers[2]->Nodes[0].activiationValue << std::endl;
    std::cout << "After Training: " << NN.Layers[2]->Nodes[1].activiationValue << std::endl;
    std::cout << "After Training: " << NN.Layers[2]->Nodes[2].activiationValue << std::endl;
    std::cout << "After Training: " << NN.Layers[2]->Nodes[3].activiationValue << std::endl;
    std::cout << "After Training: " << NN.Layers[2]->Nodes[4].activiationValue << std::endl;
    NN.Cost(expectedOutputs);
    
    std::cout << "Cost After Training: " << NN.NetworkCost << std::endl;
    return 0;
}