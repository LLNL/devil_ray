// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/vec.hpp>
#include <dray/types.hpp>
#include <dray/array.hpp>
#include <dray/GridFunction/bezier_clipping.hpp>

using namespace dray; 

TEST (dray_bezier, dray_bezier_intersect)
{
    // dray::bezier_clipping::intersect(); 
}

TEST(dray_bezier, dray_bezier_intersection_points) 
{
    // dray::bezier_clipping::intersection_points();
}

Array<Vec<Float, 3u>> makeCurve() { 
    // Make a curve with two control points. 
    Array<Vec<Float, 3u>> curve;
    curve.resize(2);

    Vec<Float, 3u> controlPointOne = {0, 0, 0}; 
    Vec<Float, 3u> controlPointTwo = {1, 1, 1}; 
    
    Vec<Float, 3u>* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = controlPointOne;
    curvePtr[1] = controlPointTwo; 
    
    return curve; 
}

TEST(dray_bezier, dray_bezier_normalized_implicit) 
{
    Array<Vec<Float, 3u>> curve = makeCurve();
    dray::bezier_clipping::NormalizedImplicitLine<Vec<Float, 3u>> line = dray::bezier_clipping::normalized_implicit<Array<Vec<Float, 3u>>, Vec<Float, 3u>>(curve); 

    Vec<Float, 3u> controlPointOne = {0, 0, 0};
    Vec<Float, 3u> controlPointTwo = {1, 1, 1}; 

    ASSERT_FLOAT_EQ(line.dist(controlPointOne), 0.);
    ASSERT_FLOAT_EQ(line.dist(controlPointTwo), 0.);
}

TEST(dray_bezier, dray_bezier_fat_line) 
{
    // Make a NormalizedImplicitLine.
    dray::bezier_clipping::NormalizedImplicitLine<Vec<Float, 3u>> line;
    line.N = {-5, 10, 5}; 
    line.P_0 = {0, 1, 2}; 

    Array<Vec<Float, 3u>> curve = makeCurve(); 

    // Compute the fat line. 
    dray::bezier_clipping::FatLine res = dray::bezier_clipping::fat_line(line, curve);
    ASSERT_FLOAT_EQ(res.upperBound, -10.);
    ASSERT_FLOAT_EQ(res.lowerBound, -20.);
}