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

typedef Vec<Float, 3u> Point;

Array<Vec<Float, 3u>> makeCurve() { 
    // Make a curve with two control points. 
    Array<Point> curve;
    curve.resize(2);

    Point controlPointOne = {0, 0, 0}; 
    Point controlPointTwo = {1, 1, 1}; 
    
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = controlPointOne;
    curvePtr[1] = controlPointTwo; 
    
    return curve; 
}

TEST (dray_bezier, dray_bezier_intersect)
{
    // dray::bezier_clipping::intersect(); 
}

TEST(dray_bezier, dray_bezier_intersection_points_two_intersections) 
{
    Array<Point> curve = makeCurve(); 
    dray::bezier_clipping::FatLine<Point> fatLine = dray::bezier_clipping::FatLine<Point>{};
    dray::bezier_clipping::NormalizedImplicitLine<Point> line = dray::bezier_clipping::NormalizedImplicitLine<Point>{};
    
    fatLine.lowerBound = 0;
    fatLine.upperBound = 1;
    line.N             = Point(); 
    line.P_0           = Point(); 
    fatLine.line       = line;

    // Takes a curve and a fatline and finds the two points of intersection (if they exist). 
    Array<Point> intersections;
    bool foundIntersections = dray::bezier_clipping::intersection_points(curve, fatLine, intersections);
    Point* intersectionsPtr = intersections.get_host_ptr(); 
    ASSERT_TRUE(foundIntersections);
    Point firstExpectedIntersection; 
    Point secondExpectedIntersection; 
    ASSERT_TRUE(intersectionsPtr[0] == firstExpectedIntersection || intersectionsPtr[1] == firstExpectedIntersection); 
    ASSERT_TRUE(intersectionsPtr[0] == secondExpectedIntersection || intersectionsPtr[1] == secondExpectedIntersection); 
}

TEST(dray_bezier, dray_bezier_intersection_points_no_intersections)
{
    Array<Point> curve = makeCurve(); 
    dray::bezier_clipping::FatLine<Point> fatLine = dray::bezier_clipping::FatLine<Point>{};
    fatLine.lowerBound = 0;
    fatLine.upperBound = 1;

    // Takes a curve and a fatline and finds the two points of intersection (if they exist). 
    Array<Point> intersections;
    bool foundIntersections = dray::bezier_clipping::intersection_points(curve, fatLine, intersections);
    ASSERT_FALSE(foundIntersections);  
}

TEST(dray_bezier, dray_bezier_normalized_implicit) 
{
    Array<Point> curve = makeCurve();
    dray::bezier_clipping::NormalizedImplicitLine<Point> line = dray::bezier_clipping::normalized_implicit<Array<Point>, Point>(curve); 

    Point controlPointOne = {0, 0, 0};
    Point controlPointTwo = {1, 1, 1}; 

    ASSERT_FLOAT_EQ(line.dist(controlPointOne), 0.);
    ASSERT_FLOAT_EQ(line.dist(controlPointTwo), 0.);
}

TEST(dray_bezier, dray_bezier_fat_line) 
{
    // Make a NormalizedImplicitLine.
    dray::bezier_clipping::NormalizedImplicitLine<Point> line;
    line.N = {-5, 10, 5}; 
    line.P_0 = {0, 1, 2}; 

    Array<Point> curve = makeCurve(); 

    // Compute the fat line. 
    dray::bezier_clipping::FatLine<Point> res = dray::bezier_clipping::fat_line(line, curve);
    ASSERT_FLOAT_EQ(res.upperBound, -10.);
    ASSERT_FLOAT_EQ(res.lowerBound, -20.);
    ASSERT_EQ(res.line.P_0, line.P_0);
    ASSERT_EQ(res.line.N, line.N);
}