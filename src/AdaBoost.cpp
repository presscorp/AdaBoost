//
//  AdaBoost.cpp
//  AdaBoost
//
//  Created by Zhalgas Baibatyr on 9/29/15.
//  Copyright (c) 2015 Zhalgas Baibatyr. All rights reserved.
//

#include "AdaBoost.h"

/*
    Constructor accepts feature values, class labels (-1 or +1) and maximum number of weak classifiers;
*/
AdaBoost::AdaBoost(const mat &features, const ivec &classes, const uint &maxWeakCount)
:   features(features),
    classes(classes),
    maxWeakCount(maxWeakCount),
    dsc(this->features, this->classes, weights),
    isModeled(false)
{
}

/*
    "train()" function trains strong classifier;
*/
void AdaBoost::train()
{
    trainErrors = vec(maxWeakCount);
    model.alpha = vec(maxWeakCount);
    model.thresholds.reserve(maxWeakCount);
    model.bounds = join_cols(min(features) - 1e-10,
                             max(features) + 1e-10);

    /* Uniform distribution of sample weights: */
    weights = ones<vec>(features.n_rows) / features.n_rows;

    /* Continuous representation of final hypothesis: */
    vec cFinalHypothesis = zeros<vec>(classes.n_rows);

    for (uint t = 0; t < maxWeakCount; ++t)
    {
        ivec weakHypothesis;
        double error;

        /*
            Acquiring the best decision stump's threshold, weak hypothesis and its error
            according to the current sample weights:
        */
        dsc.getBestStump(model.thresholds[t], weakHypothesis, error);
        ++model.nWeakCount;

        /* Stop if there is no threshold with error < 0.5  */
        if (error >= 0.5)
        {
            printf("Weak classifier error on iteration %u is reached %f.\n", t, error);

            return;
        }

        /* Calculating the weight of weak classifier for the current iteration: */
        model.alpha(t) = 0.5 * log((1 - error) / std::max(error, std::numeric_limits<double>::epsilon()));

        /* Updating the sample weights: */
        weights = weights % exp(-model.alpha(t) * conv_to<vec>::from(classes % weakHypothesis));
        weights /= sum(weights);

        /* Evaluating the training error after current iteration: */
        cFinalHypothesis += model.alpha(t) * conv_to<vec>::from(weakHypothesis);
        trainFinalHypothesis = conv_to<ivec>::from(sign(cFinalHypothesis));
        trainErrors(t) = static_cast<double>(sum(trainFinalHypothesis != classes)) / classes.n_rows;

        /* Stop if training error is reached absolute 0.00: */
        if (trainErrors(t) == 0)
        {
            printf("Training error on iteration %u is reached %f.\n", t, 0.00);
            isModeled = true;

            return;
        }
    }

    printf("Train error is %f.\n", trainErrors(maxWeakCount - 1));
    isModeled = true;

    return;
}

/*
    "test()" function tests trained strong classifier on passed data;
    To perform testing is necessary to train the "Model" first.
    Result is saved to the passed parameters "finalHypothesis" and "error";
*/
void AdaBoost::test(const mat &features, const ivec &classes,
                                               ivec &finalHypothesis, double &error)
{
    if (isModeled == false)
    {
        throw std::runtime_error("AdaBoost: Model hasn't been trained or set!\n");
    }

    /* Continuous representation of final hypothesis: */
    vec cFinalHypothesis = zeros<vec>(classes.n_rows);

    /* Evaluating weak hypotheses: */
    for (int t = 0; t < model.nWeakCount; ++t)
    {
        ivec weakHypothesis;
        dsc.classify(model.thresholds[t], features, weakHypothesis);
        cFinalHypothesis += model.alpha(t) * conv_to<vec>::from(weakHypothesis);
    }

    /* Evaluating final hypothesis and testing error: */
    finalHypothesis = conv_to<ivec>::from(sign(cFinalHypothesis));
    error = static_cast<double>(sum(finalHypothesis != classes)) / classes.n_rows;

    return;
}

/*
    "predict()" function classifies each sample by its features;
*/
void AdaBoost::predict(const mat &features, ivec &finalHypothesis)
{
    if (isModeled == false)
    {
        throw std::runtime_error("AdaBoost: Model hasn't been trained or set!\n");
    }

    /* Continuous representation of final hypothesis: */
    vec cFinalHypothesis = zeros<vec>(classes.n_rows);

    /* Evaluating weak hypotheses: */
    for (int t = 0; t < model.nWeakCount; ++t)
    {
        ivec hypothesis;
        dsc.classify(model.thresholds[t], features, hypothesis);
        cFinalHypothesis += model.alpha(t) * conv_to<vec>::from(hypothesis);
    }

    /* Predicting final hypothesis: */
    finalHypothesis = conv_to<ivec>::from(sign(cFinalHypothesis));

    return;
}

/*
    "setMaxWeakCount()" sets maximum number of weak classifiers;
*/
void AdaBoost::setMaxWeakCount(const uint &maxWeakCount)
{
    this->maxWeakCount = maxWeakCount;

    return;
}

/*
    "getModel()" provides trained or set Model;
*/
void AdaBoost::setModel(const Model &model)
{
    this->model = model;
    isModeled = true;

    return;
}

/*
    "getModel()" provides trained or set Model;
*/
void AdaBoost::getModel(Model &model) const
{
    if (isModeled == false)
    {
        throw std::runtime_error("AdaBoost: Model hasn't been trained or set!\n");
    }

    model = this->model;

    return;
}

/*
    "getTrainErrors()" function saves train errors to the passed vector;
*/
void AdaBoost::getTrainErrors(vec &trainErrors) const
{
    if (isModeled == false)
    {
        throw std::runtime_error("AdaBoost: Model hasn't been trained or set!\n");
    }

    trainErrors = this->trainErrors;

    return;
}
