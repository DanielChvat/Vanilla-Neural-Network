#pragma once
#include <Node.hpp>
#include <iostream>
#include <cstdlib>
#include <random>
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
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0, 1.0);
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
        Nodes[i].activiationValue = activationFunction(WeightedValue + Nodes[i].nodeBias);
    }
}

double Layer::activationFunction(double WeightedValue){
    return 1 / (1 + exp(-WeightedValue));
}