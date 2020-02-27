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

typedef Vec<Float, 2u> Point;
typedef Vec<Float, 3u> Vec3D; 

/*
Array<Point> makeCurve() { 
    // Make a curve with two control points. 
    Array<Point> curve;
    curve.resize(3);

    Point controlPointOne   = {0, 0}; 
    Point controlPointTwo   = {10, 5};
    Point controlPointThree = {1, 1}; 
    
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = controlPointOne;
    curvePtr[1] = controlPointTwo; 
    curvePtr[2] = controlPointThree;
    
    return curve; 
}

// ======================================
// === Normalized Implicit Line Tests ===
// ======================================
TEST(dray_bezier, dray_bezier_normalized_implicit) 
{
    // We should have a line from (0, 0) to (1, 1).
    Array<Point> curve = makeCurve();
    dray::bezier_clipping::NormalizedImplicitLine line;
    line = dray::bezier_clipping::normalized_implicit(curve); 

    Point controlPointOne = {0, 0};
    Point controlPointTwo = {1, 1}; 
    Point testPoint       = {0, 1}; 
    Float orthogonalDist  = pow((pow(0.5, 2) + pow(0.5, 2)), 0.5);

    cout << line.N << endl; 

    // Verify that the line is correct
    ASSERT_EQ(line.P_0, controlPointOne);
    ASSERT_FLOAT_EQ(dot(line.N, controlPointTwo - controlPointOne), 0.);

    // Verify that the distances are correct 
    ASSERT_FLOAT_EQ(line.dist(controlPointOne), 0.);
    ASSERT_FLOAT_EQ(line.dist(controlPointTwo), 0.);
    ASSERT_FLOAT_EQ(line.dist(testPoint), orthogonalDist); 
}

// ======================
// === Fat Line Tests ===
// ======================
TEST (dray_bezier, dray_bezier_fatline_deg_one) {
    Array<Point> curve; 
    curve.resize(2); 
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 0};
    curvePtr[1] = {1, 0};

    dray::bezier_clipping::NormalizedImplicitLine l;
    l.N = {0, 1};
    l.P_0 = {0, 0}; 

    dray::bezier_clipping::FatLine fatline;
    fatline = dray::bezier_clipping::fat_line(l, curve);

    // If we only have two points, the fat line will always have a bound of 0.
    ASSERT_FLOAT_EQ(fatline.lowerBound, 0); 
    ASSERT_FLOAT_EQ(fatline.upperBound, 0);
}

TEST (dray_bezier, dray_bezier_fatline_deg_two) {
    Array<Point> curve; 
    curve.resize(3); 
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 0};
    curvePtr[1] = {0.5, -10};
    curvePtr[2] = {1, 0}; 

    dray::bezier_clipping::NormalizedImplicitLine l;
    l.N = {0, -1};
    l.P_0 = {0, 0}; 

    dray::bezier_clipping::FatLine fatline;
    fatline = dray::bezier_clipping::fat_line(l, curve);

    // In the special case of deg = 2, we take the bounds to be half of the signed distances. 
    // Note that it may seem like the distance should be -5 instead of positive 5 but the normal vector points downwards.
    ASSERT_FLOAT_EQ(fatline.lowerBound, 0); 
    ASSERT_FLOAT_EQ(fatline.upperBound, 5);
}

TEST (dray_bezier, dray_bezier_fatline_deg_three) {
    Array<Point> curve; 
    curve.resize(4); 
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 0};
    curvePtr[1] = {3, 9};
    curvePtr[2] = {6, -18};
    curvePtr[3] = {10, 0};

    dray::bezier_clipping::NormalizedImplicitLine l;
    l.N = {0, 1};
    l.P_0 = {0, 0}; 

    dray::bezier_clipping::FatLine fatline;
    fatline = dray::bezier_clipping::fat_line(l, curve);

    // Special case where deg = 4.
    ASSERT_FLOAT_EQ(fatline.lowerBound, -8); 
    ASSERT_FLOAT_EQ(fatline.upperBound, 4);
}

TEST (dray_bezier, dray_bezier_fatline_deg_four) {
    Array<Point> curve; 
    curve.resize(5); 
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 0};
    curvePtr[1] = {2.5, 10};
    curvePtr[2] = {5, 12};
    curvePtr[3] = {7, -6};
    curvePtr[4] = {10, 0};

    dray::bezier_clipping::NormalizedImplicitLine l;
    l.N = {0, 1};
    l.P_0 = {0, 0}; 

    dray::bezier_clipping::FatLine fatline;
    fatline = dray::bezier_clipping::fat_line(l, curve);
    ASSERT_FLOAT_EQ(fatline.lowerBound, -6); 
    ASSERT_FLOAT_EQ(fatline.upperBound, 12);
}

// =================================
// === Intersection Points Tests === 
// =================================
TEST(dray_bezier, dray_bezier_intersection_points_no_intersections)
{
    Array<Point> curve; 
    curve.resize(3); 
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 10};
    curvePtr[1] = {1, 5};
    curvePtr[2] = {2, 10};

    dray::bezier_clipping::FatLine fatLine = bezier_clipping::FatLine{};
    dray::bezier_clipping::NormalizedImplicitLine normedLine;
    fatLine.lowerBound = -1;
    fatLine.upperBound = 1;
    normedLine.N = {0, 1};
    normedLine.P_0 = {0, 0};
    fatLine.line = normedLine;

    // Takes a curve and a fatline and finds the two points of intersection (if they exist). 
    // Here our fat line goes along (0, 0) -> (1, 0).
    Array<Float> intersections;
    bool foundIntersections = bezier_clipping::intersection_points(curve, fatLine, intersections);
    ASSERT_FALSE(foundIntersections);  
}

TEST(dray_bezier, dray_bezier_intersection_points_two_points) { 
    Array<Point> curve; 
    curve.resize(3);
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 20};
    curvePtr[1] = {1, 0};
    curvePtr[2] = {2, -20};

    // The points are equally spaced out, which means we have 
    // (0, 20), (0.5, 0), and (1, -20). 
    // This gives us a slope of -40 throughout. 
    dray::bezier_clipping::FatLine fatLine = bezier_clipping::FatLine{};
    dray::bezier_clipping::NormalizedImplicitLine normedLine; 
    fatLine.lowerBound = -10; 
    fatLine.upperBound = 10; 
    normedLine.N   = {0, 1}; 
    normedLine.P_0 = {0, 0}; 
    fatLine.line   = normedLine;

    Array<Float> intersections; 
    bool foundIntersections = bezier_clipping::intersection_points(curve, fatLine, intersections);
    ASSERT_TRUE(foundIntersections);  
    ASSERT_EQ(intersections.size(), 2);
    ASSERT_FLOAT_EQ(intersections.get_value(0), 0.25); 
    ASSERT_FLOAT_EQ(intersections.get_value(1), 0.75);
}

TEST(dray_bezier, dray_bezier_intersection_points_two_points_under) { 
    Array<Point> curve; 
    curve.resize(3);
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 20};
    curvePtr[1] = {1, 0};
    curvePtr[2] = {2, 20};

    // The points are equally spaced out, which means we have 
    // (0, 20), (0.5, 0), and (1, -20). 
    // This gives us a slope of -40 throughout. 
    dray::bezier_clipping::FatLine fatLine = bezier_clipping::FatLine{};
    dray::bezier_clipping::NormalizedImplicitLine normedLine; 
    fatLine.lowerBound = -10; 
    fatLine.upperBound = 10; 
    normedLine.N   = {0, 1}; 
    normedLine.P_0 = {0, 0}; 
    fatLine.line   = normedLine;

    Array<Float> intersections; 
    bool foundIntersections = bezier_clipping::intersection_points(curve, fatLine, intersections);
    ASSERT_TRUE(foundIntersections);  
    ASSERT_EQ(intersections.size(), 2);
    ASSERT_FLOAT_EQ(intersections.get_value(0), 0.25); 
    ASSERT_FLOAT_EQ(intersections.get_value(1), 0.75);
}

// ==========================
// === Intersection Tests === 
// ==========================
TEST(dray_bezier, dray_bezier_intersect_one_intersection) { 
    Array<Point> curve; 
    curve.resize(2); 
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 0};
    curvePtr[1] = {1, 0}; 

    Array<Point> curveTwo; 
    curveTwo.resize(2); 
    Point* curveTwoPtr = curveTwo.get_host_ptr(); 
    curveTwoPtr[0] = {0.5, 0.5};
    curveTwoPtr[1] = {0.5, -0.5}; 

    cout << "Before Intersect: " << curveTwoPtr[0] << " " << curveTwoPtr[1] << endl; 
    cout << "Before Intersect: " << curvePtr[0] << " " << curvePtr[1] << endl; 

    Array<Float> res;
    bool foundIntersection = dray::bezier_clipping::intersect(res, curve, curveTwo);
    
    cout << "After Intersect: " << curveTwoPtr[0] << " " << curveTwoPtr[1] << endl; 
    cout << "After Intersect: " << curvePtr[0] << " " << curvePtr[1] << endl; 
    
    Float* resPtr = res.get_host_ptr(); 
    ASSERT_TRUE(foundIntersection);
    ASSERT_EQ(res.size(), 1); 
    ASSERT_FLOAT_EQ(resPtr[0], 0.5); 

    cout << "--- NOW CHECKING THE OTHER DIRECTION ---" << endl; 
    curvePtr[0] = {0, 0}; 
    curvePtr[1] = {1, 0}; 
    curveTwoPtr[0] = {0.5, 0.5};
    curveTwoPtr[1] = {0.5, -0.5}; 

    // Now check the second curve against the first 
    Array<Float> resTwo;
    foundIntersection = dray::bezier_clipping::intersect(resTwo, curveTwo, curve); 
    Float* resTwoPtr = resTwo.get_host_ptr();
    ASSERT_TRUE(foundIntersection); 
    ASSERT_EQ(resTwo.size(), 1); 
    ASSERT_FLOAT_EQ(resTwoPtr[0], 0.5);  
    
}

TEST(dray_bezier, dray_bezier_intersect_no_intersection) { 
    Array<Point> curve; 
    curve.resize(2); 
    Point* curvePtr = curve.get_host_ptr(); 
    curvePtr[0] = {0, 0};
    curvePtr[1] = {1, 0}; 

    Array<Point> curveTwo; 
    curveTwo.resize(2); 
    Point* curveTwoPtr = curveTwo.get_host_ptr(); 
    curveTwoPtr[0] = {0, 0.5};
    curveTwoPtr[1] = {1, 0.5}; 

    Array<Float> res;
    bool foundIntersection = dray::bezier_clipping::intersect(res, curve, curveTwo);
    
    Float* resPtr = res.get_host_ptr(); 
    ASSERT_FALSE(foundIntersection);
    ASSERT_EQ(res.size(), 0); 
} */

TEST(dray_bezier, dray_bezier_multiple_iterations) { 

    // Demo: https://www.desmos.com/calculator/dqt4io0faw

    Array<Point> curve, curveTwo; 
    curve.resize(4); 
    curveTwo.resize(4);
    Point* curvePtr = curve.get_host_ptr(); 
    Point* curveTwoPtr = curveTwo.get_host_ptr(); 
    curvePtr[0] = {3, 0}; 
    curvePtr[1] = {2, 1}; 
    curvePtr[2] = {1, 1}; 
    curvePtr[3] = {0, 0}; 
    
    curveTwoPtr[0] = {1, -2.5}; 
    curveTwoPtr[1] = {3, 0}; 
    curveTwoPtr[2] = {-1, 1}; 
    curveTwoPtr[3] = {1, 2.75}; 

    Array<Float> res; 
    bool foundIntersection = dray::bezier_clipping::intersect(res, curve, curveTwo); 

    Float* resPtr = res.get_host_ptr(); 
    ASSERT_TRUE(foundIntersection); 
    ASSERT_EQ(res.size(), 1);
    cout << "THE RESULT IS: " << resPtr[0] << endl;

    // Intersection occurs at (x, y) where (0.86 < x < 0.87) and (0.61 < y < 0.62)
}

// ===================================
// === Ray-Mesh Intersection Tests === 
// ===================================

/*
TEST(dray_bezier, dray_bezier_projectTo2D) {
    size_t n = 1; 
    size_t m = 1;
    Array<Point> newControlPoints; 
    Vec3D planeOneNormal = {0, 0, 0}; 
    Vec3D planeTwoNormal = {0, 0, 0}; 
    Vec3D rayOrigin      = {0, 0, 0}; 

    Array<Vec3D> controlPoints; 
    controlPoints.resize(1); 
    Vec3D* controlPointsPtr = controlPoints.get_host_ptr(); 
    controlPointsPtr[0] = {0, 0, 0};
    
    dray::bezier_clipping::projectTo2D(
        planeOneNormal, planeTwoNormal, 
        rayOrigin, 
        controlPoints,
        newControlPoints,
        n, m);
}
*/ 