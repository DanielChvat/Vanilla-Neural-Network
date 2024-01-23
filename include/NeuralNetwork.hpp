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
        void getOutputs(double[]);
        void Cost(int[]);
        void CalculateOutputLayerNodeValues(Layer *, int[]);
        void CalculateHiddenLayerNodeValues(Layer *, Layer *);
        void UpdateNetworkGradients(double [], int []);
        void Train(trainData *, double, int);
        void Train(double [], int[], double, int);
        void UpdateGradients();
        void CreateExpected(int, int[]);
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

void NeuralNetwork::getOutputs(double inputs[]){
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
        double activationDerivative = activationFunctionDerivative(OutputLayer->Nodes[i].WeightedValue);
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
        NewNodeValue *= activationFunctionDerivative(CurrentLayer->Nodes[i].WeightedValue);
        CurrentLayer->NodeValues[i] = NewNodeValue;
    }
}

void NeuralNetwork::UpdateNetworkGradients(double inputs[], int expected[]){
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

void NeuralNetwork::CreateExpected(int label, int expected[]){
    for(int i=0; i < 10; i++){
        expected[i] = (i == label)?1: -1;
    }
}

void NeuralNetwork::Train(trainData *trainBatch, double LearningRate, int BatchSize){
    for(int i=0; i < trainBatch->ImgCount; i++){
        int expected[10];
        CreateExpected(trainBatch->data[i].label, expected);
        UpdateNetworkGradients(trainBatch->data[i].bytes, expected);
        if(i % BatchSize == 0){
            for(int i=0; i < LayerCount-1; i++){
                Layers[i]->ApplyGradients(LearningRate / BatchSize);
                Layers[i]->ResetGradients();
            }
        }
    }
}
void NeuralNetwork::Train(double trainData[], int expected[], double LearningRate, int epochs){
    for(int e=0; e < epochs; e++){
        UpdateNetworkGradients(trainData, expected);
        for(int i=0; i < LayerCount-1; i++){
            Layers[i]->ApplyGradients(LearningRate / epochs);
        }
        Cost(expected);
        std::cout << "Cost Epoch:" << e << ": " << NetworkCost << std::endl;
    }
}




