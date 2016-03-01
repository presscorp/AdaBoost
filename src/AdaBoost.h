//
//  AdaBoost.h
//  AdaBoost
//
//  Created by Zhalgas Baibatyr on 9/29/15.
//  Copyright (c) 2015 Zhalgas Baibatyr. All rights reserved.
//

#ifndef __AdaBoost__AdaBoost__
#define __AdaBoost__AdaBoost__

#include <armadillo>
#include "DSC.h"

/*
    AdaBoost algorithm for binary classification;
    Decision Stump is used as a weak classifier;
*/
class AdaBoost
{
public:

    /*
        Constructor accepts feature values, class labels (-1 or +1) and maximum number of weak classifiers;
    */
    AdaBoost(const mat &features, const ivec &classes, const uint &maxWeakCount = 100);

    /*
        "train()" function trains strong classifier;
    */
    void train();

    /*
        "test()" function tests trained strong classifier on passed data;
        To perform testing is necessary to train the "Model" first.
        Result is saved to the passed parameters "finalHypothesis" and "error";
    */
    void test(const mat &features, const ivec &classes,
                                         ivec &finalHypothesis, double &error);

    /*
        "predict()" function classifies each sample by its features;
    */
    void predict(const mat &features, ivec &labels);

    /*
        "setMaxWeakCount()" function sets maximum number of weak classifiers;
    */
    void setMaxWeakCount(const uint &maxWeakCount);

    /*
        "setModel()" function sets a new Model;
    */
    void setModel(const Model &model);

    /*
        "getModel()" function provides trained or set Model;
    */
    void getModel(Model &model) const;

    /*
        "getTrainErrors()" function saves train errors to the passed vector;
    */
    void getTrainErrors(vec &trainErrors) const;

private:
    const mat features;
    const ivec classes;
    uint maxWeakCount;
    vec weights;
    DSC dsc;

    bool isModeled;
    Model model;
    ivec trainFinalHypothesis;
    vec trainErrors;
};

#endif /* defined(__AdaBoost__AdaBoost__) */
