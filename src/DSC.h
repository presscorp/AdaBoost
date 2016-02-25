//
//  DSC.h
//  AdaBoost
//
//  Created by Zhalgas Baibatyr on 9/30/15.
//  Copyright (c) 2015 Zhalgas Baibatyr. All rights reserved.
//

#ifndef __AdaBoost__DSC__
#define __AdaBoost__DSC__

#include <armadillo>
#include "Model.h"

/*
    Decision Stump Classifier designed for AdaBoost;
*/
class DSC
{
public:

    /*
        Constructor accepts feature values, class labels (-1 or +1) and sample weights;
        Passed data will be stored as a reference to original data;
    */
    DSC(const arma::mat &features,
        const arma::ivec &classes,
        const arma::vec &weights,
        const int &nThresholds = 1e+5);

    /*
        "getBestStump()" function saves the best decision stump's threshold, weak hypothesis and its error;
    */
    void getBestStump(Threshold &threshold, ivec &hypothesis, double &error);

    /*
        "classify()" function evaluates hypothesis for the given threshold on specific feature;
    */
    void classify(const Threshold &threshold, const mat &features, ivec &hypothesis);

private:
    const int N_THRESHOLDS;
    const mat &features;
    const ivec &classes;
    const vec &weights;
    vec class1Thresholds;
    vec class2Thresholds;
    vec featureIndexes1;
    vec featureIndexes2;
    rowvec minFeatures;
    rowvec maxFeatures;
    rowvec ranges;
};

#endif /* defined(__AdaBoost__DSC__) */
