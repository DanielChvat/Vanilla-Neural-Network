#pragma once
#include <Layer.hpp>
#include <stdio.h>
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
        void Train(trainData *, double, int, int);
        void Train(trainData *, double, int, int, int []);
        void Train(double [], int[], double, int);
        double getAccuracy(trainData *);
        int getLabel(img);
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

void NeuralNetwork::Train(trainData *trainBatch, double LearningRate, int BatchSize, int epochs){
    for(int e = 0; e < epochs; e++){
        double totalCost = 0;
        for(int i=0; i < trainBatch->ImgCount; i++){
            int expected[10];
            CreateExpected(trainBatch->data[i].label, expected);
            UpdateNetworkGradients(trainBatch->data[i].bytes, expected);
            if(i % BatchSize == 0 && i != 0){
                for(int j=0; j < LayerCount-1; j++){
                    Layers[j]->ApplyGradients(LearningRate / (BatchSize * epochs));
                    Layers[j]->ResetGradients();
                }
            }
        }
        for(int k=0; k < trainBatch->ImgCount; k++){
            getOutputs(trainBatch->data[k].bytes);
            int expected[10];
            CreateExpected(trainBatch->data[k].label, expected);
            Cost(expected);
            totalCost += NetworkCost;
        }
        printf("Epoch %d: %lf\n", e, totalCost/(double)trainBatch->ImgCount);
        //std::cout << "Epoch:" << e << ": " << totalCost/((e+1) * trainBatch->ImgCount) << "\n";
    }
}

void NeuralNetwork::Train(trainData *trainBatch, double LearningRate, int BatchSize, int epochs, int expectedOutputs[]){
    for(int e = 0; e < epochs; e++){
        for(int i=0; i < trainBatch->ImgCount; i++){
            int expected[10];
            CreateExpected(trainBatch->data[i].label, expected);
            UpdateNetworkGradients(trainBatch->data[i].bytes, expected);
            if(i % BatchSize == 0 && i != 0){
                for(int j=0; j < LayerCount-1; j++){
                    Layers[j]->ApplyGradients(LearningRate / (BatchSize * epochs));
                    Layers[j]->ResetGradients();
                }
            }
        }
        getOutputs(trainBatch->data[73].bytes);
        Cost(expectedOutputs);
        printf("Epoch:%d:%lf\n", e, NetworkCost);
    }
}
void NeuralNetwork::Train(double trainData[], int expected[], double LearningRate, int epochs){
    for(int e=0; e < epochs; e++){
        UpdateNetworkGradients(trainData, expected);
        for(int i=0; i < LayerCount-1; i++){
            Layers[i]->ApplyGradients(LearningRate / epochs);
        }
        Cost(expected);
        std::cout << "Cost Epoch:" << e << ": " << NetworkCost << "\n";
    }
}

int NeuralNetwork::getLabel(img data){
    getOutputs(data.bytes);
    Layer *outputLayer = Layers[LayerCount-1];
    double outputs[outputLayer->NodeCount];
    for(int i = 0; i < outputLayer->NodeCount; i++){
        outputs[i] = outputLayer->Nodes[i].activiationValue;
    }

    int label = 0;
    double m = outputs[label];

    for(int i=1; i < outputLayer->NodeCount; i++){
        if(outputs[i] > m){
            label = i;
            m = outputs[i];
        }
    }
    return label;
}

double NeuralNetwork::getAccuracy(trainData *testData){
    int correct = 0;
    for(int i=0; i < testData->ImgCount; i++){
        int actual = testData->data[i].label;
        int predicted = getLabel(testData->data[i]);
        if(actual == predicted){
            printf("Image %d: Correct\n", i);
            correct++;
        }else{
            printf("Image %d: Incorrect\n", i);
        }
    }
    return correct / (double)testData->ImgCount; 
}




