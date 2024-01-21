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
        void CalculateOutputLayerNodeValues(Layer *, int[]);
        void CalculateHiddenLayerNodeValues(Layer *, Layer *);
        void UpdateNetworkGradients(int [], int []);
        void Train(int [], int [], double);
        void UpdateGradients();
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



void NeuralNetwork::CalculateOutputLayerNodeValues(Layer *OutputLayer, int expected[]){
    for(int i=0; i < OutputLayer->NodeCount; i++){
        double costDerivative = costFunctionDerivative(OutputLayer->Nodes[i].activiationValue, expected[i]);
        double activationDerivative = activationFunctionDerivative(OutputLayer->Nodes[i].activiationValue);
        OutputLayer->NodeValues[i] = activationDerivative * costDerivative;
    }
}

void NeuralNetwork::CalculateHiddenLayerNodeValues(Layer *CurrentLayer, Layer *NextLayer){
    for(int i=0; i < CurrentLayer->NodeCount; i++){
        double NewNodeValue = 0;
        for(int j=0; j < NextLayer->NodeCount; j++){
            double oldNodeValue = NextLayer->NodeValues[j];
            NewNodeValue +=  oldNodeValue * CurrentLayer->NodeWeights[i][j];
        }
        NewNodeValue *= activationFunctionDerivative(CurrentLayer->Nodes[i].activiationValue);
        CurrentLayer->NodeValues[i] = NewNodeValue;
    }
}

void NeuralNetwork::UpdateNetworkGradients(int inputs[], int expected[]){
    Layer *OutputLayer = Layers[LayerCount-1];
    getOutputs(inputs);
    CalculateOutputLayerNodeValues(OutputLayer, expected);
    OutputLayer->UpdateGradients(Layers[LayerCount-2]);

    for(int i = LayerCount - 2; i >= 1; i--){
        Layer *HiddenLayer = Layers[i];
        Layer *prevLayer = Layers[i-1];
        Layer *NextLayer = Layers[i+1];
        CalculateHiddenLayerNodeValues(HiddenLayer, NextLayer);
        HiddenLayer->UpdateGradients(prevLayer);
    }
}

void NeuralNetwork::Train(int trainData[], int expected[], double LearningRate){
    UpdateNetworkGradients(trainData, expected);
    for(int i=0; i < LayerCount; i++){
        Layers[i]->ApplyGradients(LearningRate);
    }
}




