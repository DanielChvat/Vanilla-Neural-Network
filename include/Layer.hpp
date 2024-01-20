#pragma once
#include <Node.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>


class Layer{
    public:
        Node *Nodes;
        float **NodeWeights;
        int NodeCount;
        int NextNodeCount;
        ~Layer();
        Layer(int, int);
        Layer();
        void init();
        void calculateLayerOutputs(Layer *);
        double activationFunction(double);
};

Layer::Layer(int NodeCount, int NextNodeCount): NodeCount(NodeCount), NextNodeCount(NextNodeCount) {
    Nodes = new Node[NodeCount];
    if(NextNodeCount){
        NodeWeights = new float*[NodeCount];
        for(int i=0; i < NodeCount; i++){
            NodeWeights[i] = new float[NextNodeCount];
        }
    }
    init();
}

Layer::Layer(){}

Layer::~Layer(){
    delete []Nodes;
    if(NextNodeCount){
        for(int i=0; i < NodeCount; i++){
            delete []NodeWeights[i];
        }
        delete []NodeWeights;
    }
}

void Layer::init(){
    for(int i=0; i < NodeCount; i++){
        //Nodes[i].nodeBias = (rand() % 101) / 100.0f;
        Nodes[i].nodeBias = 0.5f;
        for(int j=0; j < NextNodeCount; j++){
            //NodeWeights[i][j] = (rand() % 101) / 100.0f;
            NodeWeights[i][j] = 0.5f;
        }
    }
}

void Layer::calculateLayerOutputs(Layer *prevLayer){
    double WeightedValue;
    for(int i=0; i < NodeCount; i++){
        WeightedValue = 0;
        for(int j=0; j < prevLayer->NodeCount; j++){
            WeightedValue += prevLayer->NodeWeights[j][i] * prevLayer->Nodes[j].activiationValue;
        }
        Nodes[i].activiationValue = activationFunction(WeightedValue + Nodes[i].nodeBias);
    }
}

double Layer::activationFunction(double WeightedValue){
    return 1 / (1 + exp(-WeightedValue));
}