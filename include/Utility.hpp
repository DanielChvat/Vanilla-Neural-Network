#pragma once
#include <fstream>
#include <string>
#include <cmath>
#include <Layer.hpp>
#include <sstream>

typedef 
struct img{
    char label;
    int bytes[784];
} img;

class trainData{
    public:
        img *data;
        int ImgCount;
        std::fstream file;
        std::string filePath;
        trainData(std::string);
        trainData(std::string, int);
        ~trainData();
        void getImgCount();
};


trainData::trainData(std::string filePath) : ImgCount(0), filePath(filePath){
    getImgCount();
    std::string line, token, headers;
    file.open(filePath, std::ios::in);
    data = new img[ImgCount];
    getline(file, headers);

    int i = 0;
    while(file >> line){
        data[i].label = line[0] - '0';
        line.erase(0, 2);
        std::stringstream input(line);
        int j = 0;
        while(getline(input, token, ',')){
            data[i].bytes[j++] = stoi(token);
        }
        i++;
    }
    file.close();
}

trainData::trainData(std::string filePath, int dataSize) : ImgCount(0), filePath(filePath){
    getImgCount();
    std::string line, token, headers;
    file.open(filePath, std::ios::in);

    getline(file, headers);

    ImgCount = (dataSize < ImgCount)?dataSize: ImgCount;
    data = new img[ImgCount];

    for(int i=0; i < dataSize; i++){
        file >> line;
        data[i].label = line[0] - '0';
        line.erase(0, 2);
        std::stringstream input(line);
        int j = 0;
        while(getline(input, token, ',')){
            data[i].bytes[j++] = stoi(token);
        }
    }
    file.close();
}

trainData::~trainData(){
    delete []data;
}

void trainData::getImgCount(){
    file.open(filePath, std::ios::in);
    std::string line;
    while(file >> line)ImgCount++;
    ImgCount--;
    file.close();
}

double activationFunction(double W){
    return 1 / (1 + exp(-W));
}

double activationFunctionDerivative(double activationValue){
    return activationValue * (1 - activationValue);
}

double costFunction(Layer *OutputLayer, int expected[]){
    double cost = 0;
    for(int i=0; i < OutputLayer->NodeCount; i++){
        double difference = (OutputLayer->Nodes[i].activiationValue - expected[i]);
        cost += difference * difference;
    }
    return cost;
}

double costFunctionDerivative(double activationValue, double expected){
    return 2 * (activationValue - expected);
}
