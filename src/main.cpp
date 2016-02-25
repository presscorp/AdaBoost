//
//  main.cpp
//  AdaBoost
//
//  Created by Zhalgas Baibatyr on 10/1/14.
//  Copyright (c) 2014 Zhalgas Baibatyr. All rights reserved.
//

#include "AdaBoost.h"

int main()
{
    /* Training data: */
    arma::mat features;
    features.load("../data/train/features.dat");
    arma::ivec classes;
    classes.load("../data/train/classes.dat");

    /* Feeding our AdaBoost algorithm with data: */
    AdaBoost adaboost(features, classes);

    /* Launch training: */
    adaboost.train();


    /* Testing data: */
    arma::mat features2;
    features2.load("../data/test/features.dat");
    arma::ivec classes2;
    classes2.load("../data/test/classes.dat");

    /* Test results to obtain: */
    ivec finalHypothesis;
    double error;

    /* Perform testing: */
    adaboost.test(features2, classes2, finalHypothesis, error);

    printf("Testing error is %f.\n", error);

    return EXIT_SUCCESS;
}
