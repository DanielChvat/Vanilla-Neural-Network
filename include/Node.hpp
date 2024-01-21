#pragma once

class Node{
    public:
    double WeightedValue;
    double activiationValue;
    double nodeBias;
    Node();
};

Node::Node(): activiationValue(0), nodeBias(0){}