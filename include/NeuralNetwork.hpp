#pragma once
#include <Layer.hpp>
#include <iostream>
#include <Utility.hpp>

extern "C++" double costFunction(Layer *, int);
extern "C++" double costFunctionDerivative(double, double);

class NeuralNetwork{
    public:
        Layer **Layers;
        int LayerCount; 
        double NetworkCost;
        NeuralNetwork(int, int []);
        ~NeuralNetwork();
        void getOutputs(int []);
        void Cost(int[]);
        void CalculateOutputLayerNodeValues(int[]);
};

NeuralNetwork::NeuralNetwork(int LayerCount, int LayerNodeCounts[]): LayerCount(LayerCount), NetworkCost(0) {
    Layers = new Layer*[LayerCount];
    for(int i=0; i < LayerCount; i++){
        if(i != (LayerCount - 1)) Layers[i] = new Layer(LayerNodeCounts[i], LayerNodeCounts[i+1]);
        else Layers[i] = new Layer(LayerNodeCounts[i], 0);
    }
}

NeuralNetwork::~NeuralNetwork(){

    for(int i=0; i < LayerCount; i++){
        delete Layers[i];
    }
    delete []Layers;
}

void NeuralNetwork::getOutputs(int inputs[]){
    for(int i=0; i < Layers[0]->NodeCount; i++){
        Layers[0]->Nodes[i].activiationValue = inputs[i];
    }

    for(int n=1; n < LayerCount; n++){
        Layers[n]->calculateLayerOutputs(Layers[n-1]);
    }
}

void NeuralNetwork::Cost(int expected[]){
    NetworkCost = costFunction(Layers[LayerCount-1], expected);
}



void NeuralNetwork::CalculateOutputLayerNodeValues(int expected[]){
    Layer *OutputLayer = Layers[LayerCount-1];

    for(int i=0; i < OutputLayer->NodeCount; i++){
        double costDerivative = costFunctionDerivative(OutputLayer->Nodes[i].activiationValue, expected[i]);
        double activationDerivative = activationFunctionDerivative(OutputLayer->Nodes[i].activiationValue);
        OutputLayer->NodeValues[i] = activationDerivative * costDerivative;
        std::cout << "NodeValue Node: " << i << " " << OutputLayer->NodeValues[i] << std::endl;
    }
}

