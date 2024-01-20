#pragma once
#include <Layer.hpp>
#include <Utility.hpp>

class NeuralNetwork{
    public:
        Layer **Layers;
        double ***WeightGradients;
        int LayerCount; 
        double NetworkCost;
        NeuralNetwork(int, int []);
        ~NeuralNetwork();
        void getOutputs(int []);
        void getCost(int[]);
        void Learn(trainData * const, int);
};

NeuralNetwork::NeuralNetwork(int LayerCount, int LayerNodeCounts[]): LayerCount(LayerCount), NetworkCost(0) {
    Layers = new Layer*[LayerCount];
    WeightGradients = new double**[LayerCount];
    for(int i=0; i < LayerCount; i++){
        if(i != (LayerCount - 1)) Layers[i] = new Layer(LayerNodeCounts[i], LayerNodeCounts[i+1]);
        else Layers[i] = new Layer(LayerNodeCounts[i], 0);
        WeightGradients[i] = new double*[Layers[i]->NodeCount];
        for(int j=0; j < Layers[i]->NodeCount; j++){
            WeightGradients[i][j] = new double[Layers[i]->NextNodeCount];
        }
    }
}

NeuralNetwork::~NeuralNetwork(){
    for(int i=0; i < LayerCount; i++){
        for(int j=0; j < Layers[i]->NodeCount; j++){
            delete []WeightGradients[i][j];
        }
        delete [] WeightGradients[i];

        delete Layers[i];
    }
    delete []WeightGradients;
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

void NeuralNetwork::getCost(int expected[]){
    for(int i=0; i < Layers[LayerCount-1]->NodeCount; i++){
        double difference = (Layers[LayerCount-1]->Nodes[i].activiationValue - expected[i]);
        NetworkCost += difference * difference;
    }
}

void Learn(trainData * const LearnData, int BatchSize){
    
}