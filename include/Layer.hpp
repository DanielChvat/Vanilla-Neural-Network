#pragma once
#include <Node.hpp>
#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>
#include <Utility.hpp>
#include <cmath>

extern "C++" double activationFunction(double);

class Layer{
    public:
        Node *Nodes;
        double **NodeWeights;
        double **WeightGradients;
        double *NodeValues;
        double *BiasGradients;
        int NodeCount;
        int NextNodeCount;
        ~Layer();
        Layer(int, int);
        Layer();
        void init();
        void calculateLayerOutputs(Layer *);
        void UpdateGradients(Layer *);
        void ApplyGradients(double);
};

Layer::Layer(int NodeCount, int NextNodeCount): NodeCount(NodeCount), NextNodeCount(NextNodeCount) {
    Nodes = new Node[NodeCount];
    NodeValues = new double[NodeCount];
    BiasGradients = new double[NodeCount];
    if(NextNodeCount){
        WeightGradients = new double*[NodeCount];
        NodeWeights = new double*[NodeCount];
        for(int i=0; i < NodeCount; i++){
            WeightGradients[i] = new double[NextNodeCount];
            NodeWeights[i] = new double[NextNodeCount];
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
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for(int i=0; i < NodeCount; i++){
        //Nodes[i].nodeBias = dist(gen);
        Nodes[i].nodeBias = 1;
        //Nodes[i].nodeBias = 0.00005f;
        BiasGradients[i] = 0;
        for(int j=0; j < NextNodeCount; j++){
            NodeWeights[i][j] = dist(gen);
            //NodeWeights[i][j] = 0.00005f;
            WeightGradients[i][j] = 0;
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

void Layer::UpdateGradients(Layer *prevLayer){
    for(int i=0; i < NodeCount; i++){
        for(int j=0; j < prevLayer->NodeCount; j++){
            double costDerivativewrtWeight = prevLayer->Nodes[j].activiationValue * NodeValues[i];
            prevLayer->WeightGradients[j][i] += costDerivativewrtWeight;
        }
        double costDerivativewrtBias = NodeValues[i];
        BiasGradients[i] += costDerivativewrtBias;
    }
}

void Layer::ApplyGradients(double LearningRate){
    for(int i=0; i < NodeCount; i++){
        for(int j = 0; j < NextNodeCount; j++){
            NodeWeights[i][j] -= WeightGradients[i][j] * LearningRate;
        }
        Nodes[i].nodeBias -= BiasGradients[i] * LearningRate;
    }
}
