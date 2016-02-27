//
//  DSC.cpp
//  AdaBoost
//
//  Created by Zhalgas Baibatyr on 9/30/15.
//  Copyright (c) 2015 Zhalgas Baibatyr. All rights reserved.
//

#include "DSC.h"
#include "functions.h"

/*
    Constructor accepts feature values, class labels (-1 or +1) and sample weights;
    Passed data will be stored as a reference to original data;
*/
DSC::DSC(const mat &features, const ivec &classes, const vec &weights, const int &nThresholds)
:   N_THRESHOLDS(nThresholds),
    features(features),
    classes(classes),
    weights(weights)
{
    /* Separating features by classes: */
    mat class1Features = features.rows(find(classes == -1));
    mat class2Features = features.rows(find(classes == +1));

    /* Determining the ranges for each feature: */
    minFeatures = min(features) - 1e-10;
    maxFeatures = max(features) + 1e-10;
    ranges = maxFeatures - minFeatures;

    /* Distribution of threshold values for each feature; */
    mat class1ThresholdsMat = floor(((class1Features.each_row() - minFeatures).each_row() / ranges) * (N_THRESHOLDS - 1) + 1 - 1e-10);
    class1ThresholdsMat(find(class1ThresholdsMat < 0)).zeros();
    mat class2ThresholdsMat = ceil(((class2Features.each_row() - minFeatures).each_row() / ranges) * (N_THRESHOLDS - 1) + 1 + 1e-10);
    class2ThresholdsMat(find(class2ThresholdsMat > N_THRESHOLDS - 1)).fill(N_THRESHOLDS - 1);

    /* Threshold values is vectorized by column: */
    class1Thresholds = vectorise(class1ThresholdsMat);
    class2Thresholds = vectorise(class2ThresholdsMat);

    /*
        Providing indexes of features for each label;
        After that matrix is vectorized by column:
    */
    featureIndexes1 = vectorise(repmat(linspace<rowvec>(0, features.n_cols - 1, features.n_cols), class1Features.n_rows, 1));
    featureIndexes2 = vectorise(repmat(linspace<rowvec>(0, features.n_cols - 1, features.n_cols), class2Features.n_rows, 1));
}

/*
    "getBestStump()" function saves the best decision stump's threshold, weak hypothesis and its error;
*/
void DSC::getBestStump(Threshold &threshold, ivec &hypothesis, double &error)
{
    /* Separating weights by classes: */
    mat class1Weights = weights(find(classes == -1));
    mat class2Weights = weights(find(classes == +1));

    /* Accumulate weights for given thresholds ("number of thresholds" x "number of features"): */
    mat thresholdWeights1 = accumarray(join_rows(class1Thresholds, featureIndexes1),
                                       repmat(class1Weights, features.n_cols, 1),
                                       SizeMat(N_THRESHOLDS, features.n_cols));
    mat thresholdWeights2 = accumarray(join_rows(class2Thresholds, featureIndexes2),
                                       repmat(class2Weights, features.n_cols, 1),
                                       SizeMat(N_THRESHOLDS, features.n_cols));

    /*
        Looking for threshold with minimum error;
        Here we construct cummulative sum of weights for seaprate classes, then we add them;
        In order to find the smallest error, we consider both directions of a threshold;
    */
    mat thresholdErrors = flipud(cumsum(flipud(thresholdWeights1))) + cumsum(thresholdWeights2);
    thresholdErrors = join_cols(thresholdErrors, sum(weights) - thresholdErrors);
    uword index;
    uword featureType;
    error = thresholdErrors.min(index, featureType);
    threshold.featureType = featureType;
    if (index > N_THRESHOLDS - 1)
    {
        threshold.direction = -1;
        index -= N_THRESHOLDS;
    }
    else
    {
        threshold.direction = +1;
    }

    threshold.value = minFeatures(featureType) + (ranges(featureType) / N_THRESHOLDS) * index;

    /* Evaluating hypothesis for derived threshold: */
    classify(threshold, features, hypothesis);

    return;
}

/* Hypothesis evaluation using specified thershold on the given feature: */
void DSC::classify(const Threshold &threshold, const mat &features, ivec &hypothesis)
{
    /* Set +1 label if selected feature value satisfies threshold direction, ... */
    if (threshold.direction == +1)
    {
        hypothesis = conv_to<ivec>::from((features.col(threshold.featureType) >= threshold.value));
    }
    else /* if (threshold.direction == -1) */
    {
        hypothesis = conv_to<ivec>::from((features.col(threshold.featureType) <= threshold.value));
    }

    /* ... otherwise set to -1: */
    hypothesis(find(hypothesis == 0)).fill(-1);

    return;
}
