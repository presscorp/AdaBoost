//
//  Model.h
//  AdaBoost
//
//  Created by Zhalgas Baibatyr on 9/30/15.
//  Copyright (c) 2015 Zhalgas Baibatyr. All rights reserved.
//

#ifndef AdaBoost_Model_h
#define AdaBoost_Model_h

#include <armadillo>

using namespace arma;

/*
    Threshold data for the single feature:
*/
struct Threshold
{
    /*
        Feature index;
    */
    uword featureType;

    /*
        Threashold value;
    */
    double value;

    /*
        Direction values: -1 , +1;
        Don't confuse with class labels;
    */
    shword direction;
};

/*
    Model acquired after the training:
*/
struct Model
{
    /*
        Number of weak classifiers;
    */
    uint nWeakCount;

    /*
        Boundaries of each feature represented as matrix:
        [ minFeature1, minFeature2, minFeature3, ... ;
          maxFeature1, maxFeature2, maxFeature3, ... ; ]
    */
    mat bounds;

    /*
        Weights for each weak hypothesis;
    */
    vec alpha;

    /*
        Thresholds' data;
    */
    std::vector<Threshold> thresholds;
};

#endif
