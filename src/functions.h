//
//  functions.h
//  AdaBoost
//
//  Created by Zhalgas Baibatyr on 10/3/15.
//  Copyright (c) 2015 Zhalgas Baibatyr. All rights reserved.
//

#ifndef AdaBoost_functions_h
#define AdaBoost_functions_h

#include <armadillo>

using namespace arma;

enum { ACCUMARRAY_SUM };

/*
    Construct matrix with value accumulation;
    "subs" - subscripts matrix (Size: m x 2)
    "val"  - values vector
    "sz"   - size of the output matrix
*/
mat accumarray(const mat &subs,
               const vec &val,
               const SizeMat &sz,
               const int &fun = ACCUMARRAY_SUM,
               const double &fillval = 0)
{
    mat output(sz);
    output.fill(fillval);

    if (fun == ACCUMARRAY_SUM)
    {
        for (uint i = 0; i < val.n_rows; ++i)
        {
            output(subs(i, 0), subs(i, 1)) += val(i);
        }
    }
    else
    {
        throw std::runtime_error("Improper \"fun\" parameter in accumarray() function!\n");
    }

    return output;
}

#endif
