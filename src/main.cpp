#include <iostream>
#include <cstdlib>
#include <ctime>
#include <Utility.hpp>
#include <NeuralNetwork.hpp>

int main(void){
    srand(time(0));
    
    int nLayers = 3;
    int LayerNodeCounts[] = {754, 1000, 10};
    int expectedOutputs[] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};

    NeuralNetwork NN(nLayers, LayerNodeCounts);

    std::string filePath = "E:/Programming/NeuralNetworks/dataset/mnist_train.csv";
    trainData data(filePath, 1);

    NN.getOutputs(data.data[0].bytes);

    std::cout << NN.Layers[2]->Nodes[0].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[1].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[2].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[3].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[4].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[5].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[6].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[7].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[8].activiationValue << std::endl;
    std::cout << NN.Layers[2]->Nodes[9].activiationValue << std::endl;

    NN.getCost(expectedOutputs);
    
    std::cout << NN.NetworkCost << std::endl;

    return 0;
}