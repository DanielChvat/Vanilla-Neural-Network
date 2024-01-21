#pragma once
#include <Node.hpp>
#include <iostream>
#include <cstdlib>
#include <random>
#include <Utility.hpp>
#include <cmath>

extern "C++" double activationFunction(double);

class Layer{
    public:
        Node *Nodes;
        float **NodeWeights;
        float **WeightGradients;
        float *NodeValues;
        float *BiasGradients;
        int NodeCount;
        int NextNodeCount;
        ~Layer();
        Layer(int, int);
        Layer();
        void init();
        void calculateLayerOutputs(Layer *);
};

Layer::Layer(int NodeCount, int NextNodeCount): NodeCount(NodeCount), NextNodeCount(NextNodeCount) {
    Nodes = new Node[NodeCount];
    NodeValues = new float[NodeCount];
    if(NextNodeCount){
        BiasGradients = new float[NodeCount];
        WeightGradients = new float*[NodeCount];
        NodeWeights = new float*[NodeCount];
        for(int i=0; i < NodeCount; i++){
            WeightGradients[i] = new float[NextNodeCount];
            NodeWeights[i] = new float[NextNodeCount];
        }
    }
    init();
}

Layer::Layer(){}

Layer::~Layer(){
    delete []Nodes;
    delete []NodeValues;
    if(NextNodeCount){
        delete []BiasGradients;
        for(int i=0; i < NodeCount; i++){
            delete []NodeWeights[i];
            delete []WeightGradients[i];
        }
        delete []WeightGradients;
        delete []NodeWeights;
    }
}

void Layer::init(){
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0, 0.0001);
    for(int i=0; i < NodeCount; i++){
        Nodes[i].nodeBias = dist(gen);
        //Nodes[i].nodeBias = 0.00005f;
        for(int j=0; j < NextNodeCount; j++){
            NodeWeights[i][j] = dist(gen);
            //NodeWeights[i][j] = 0.00005f;
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
        Nodes[i].WeightedValue = WeightedValue + Nodes[i].nodeBias;
        Nodes[i].activiationValue = activationFunction(Nodes[i].WeightedValue);
    }
}
