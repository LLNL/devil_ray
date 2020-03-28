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
 
dray::bezier_clipping::Curve<2> makeCurve() { 
    // Make a curve with two control points. 
    dray::bezier_clipping::Curve<2> curve;

    Point controlPointOne   = {0, 0}; 
    Point controlPointTwo   = {10, 5};
    Point controlPointThree = {1, 1}; 
    
    Vec<Point, 3u> points = {controlPointOne, controlPointTwo, controlPointThree};

    uint32 i = 0; 
    for (auto &point : curve.components()) { 
        point = points[i++];
    }
    
    return curve; 
}

// ======================================
// === Normalized Implicit Line Tests ===
// ======================================

TEST(dray_bezier, dray_bezier_normalized_implicit) 
{
    // We should have a line from (0, 0) to (1, 1).
    dray::bezier_clipping::Curve<2> curve = makeCurve();
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
    dray::bezier_clipping::Curve<1> curve; 

    uint32 i = 0;
    for (auto &pt : curve.components()) {
        if (i == 0) {
            pt = {0, 0}; 
            ++i; 
        } else {
            pt = {1, 0};
        }
    } 

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
    dray::bezier_clipping::Curve<2> curve; 
 
    Point p1 = {0, 0};
    Point p2 = {0.5, -10};
    Point p3 = {1, 0}; 

    Vec<Point, 3u> ptArray = {p1, p2, p3};

    uint32 i = 0;
    for (auto &pt : curve.components()) { 
        pt = ptArray[i++];
    }

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
    dray::bezier_clipping::Curve<3> curve; 

    Point p1 = {0, 0};
    Point p2 = {3, 9};
    Point p3 = {6, -18};
    Point p4 = {10, 0};

    Vec<Point, 4u> points = {p1, p2, p3, p4};
    uint32 i = 0;
    for (auto &pt : curve.components())
        pt = points[i++];

    dray::bezier_clipping::NormalizedImplicitLine l;
    l.N = {0, 1};
    l.P_0 = {0, 0}; 

    dray::bezier_clipping::FatLine fatline;
    fatline = dray::bezier_clipping::fat_line(l, curve);

    // Special case where deg = 4.
    ASSERT_FLOAT_EQ(fatline.lowerBound, -8); 
    ASSERT_FLOAT_EQ(fatline.upperBound, 4);
}

// ====== PASSING UP TO HERE ==========

TEST (dray_bezier, dray_bezier_fatline_deg_four) {
    dray::bezier_clipping::Curve<4> curve; 

    Point p1 = {0, 0};
    Point p2 = {2.5, 10};
    Point p3 = {5, 12};
    Point p4 = {7, -6};
    Point p5 = {10, 0};

    Vec<Point, 5u> points = {p1, p2, p3, p4, p5}; 
    uint32 i = 0; 
    for (auto &pt : curve.components()) {
        pt = points[i++];
    }

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
    dray::bezier_clipping::Curve<2> curve; 
    
    Point p1 = {0, 10};
    Point p2 = {1, 5};
    Point p3 = {2, 10};
    Vec<Point, 3u> points = {p1, p2, p3};

    uint32 i = 0; 
    for (auto &pt : curve.components()) {
        pt = points[i++];
    }

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
    dray::bezier_clipping::Curve<2> curve; 

    Point p1 = {0, 20};
    Point p2 = {1, 0};
    Point p3 = {2, -20};

    Vec<Point, 3u> points = {p1, p2, p3};

    uint32 i = 0; 
    for (auto &pt : curve.components()) {
        pt = points[i++];
    }


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
    dray::bezier_clipping::Curve<2> curve; 
    
    Point p0 = {0, 20};
    Point p1 = {1, 0};
    Point p2 = {2, 20};

    Vec<Point, 3u> points = {p0, p1, p2};

    uint32 i = 0;
    for (auto &pt : curve.components())
        pt = points[i++];

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
    dray::bezier_clipping::Curve<1> curve; 
    Point p0 = {0, 0};
    Point p1 = {1, 0}; 

    dray::bezier_clipping::Curve<1> curveTwo; 
    Point p2 = {0.5, 0.5};
    Point p3 = {0.5, -0.5}; 

    Vec<Point, 2u> points = {p0, p1};
    Vec<Point, 2u> pointsTwo = {p2, p3};

    uint32 i = 0; 
    for (auto &pt : curve.components())
        pt = points[i++];
    
    i = 0;
    for (auto &pt : curveTwo.components())
        pt = pointsTwo[i++];

    Array<Float> res;
    bool foundIntersection = dray::bezier_clipping::intersect(res, curve, curveTwo, false);

    Float* resPtr = res.get_host_ptr(); 
    ASSERT_EQ(res.size(), 1); 
    ASSERT_TRUE(foundIntersection);
    ASSERT_FLOAT_EQ(resPtr[0], 0.5); 

    // Now check the second curve against the first 
    Array<Float> resTwo;
    foundIntersection = dray::bezier_clipping::intersect(resTwo, curveTwo, curve, false); 
    Float* resTwoPtr = resTwo.get_host_ptr();
    ASSERT_TRUE(foundIntersection); 
    ASSERT_EQ(resTwo.size(), 1); 
    ASSERT_FLOAT_EQ(resTwoPtr[0], 0.5);  
}

TEST(dray_bezier, dray_bezier_intersect_no_intersection) { 
    dray::bezier_clipping::Curve<1> curve; 
    Point p0 = {0, 0};
    Point p1 = {1, 0}; 

    dray::bezier_clipping::Curve<1> curveTwo;
    Point p2 = {0, 0.5};
    Point p3 = {1, 0.5}; 

    Vec<Point, 2u> points    = {p0, p1};
    Vec<Point, 2u> pointsTwo = {p2, p3};

    uint32 i = 0;
    for (auto &pt : curve.components())
        pt = points[i++];
    
    i = 0;
    for (auto &pt : curve.components())
        pt = pointsTwo[i++];
    

    Array<Float> res;
    bool foundIntersection = dray::bezier_clipping::intersect(res, curve, curveTwo, false);
    
    Float* resPtr = res.get_host_ptr(); 
    ASSERT_FALSE(foundIntersection);
    ASSERT_EQ(res.size(), 0); 
} 

TEST(dray_bezier, dray_bezier_multiple_iterations) { 

    // Demo: https://www.desmos.com/calculator/dqt4io0faw

    Vec<Float, 2u> p1 = {3, 0}; 
    Vec<Float, 2u> p2 = {2, 1}; 
    Vec<Float, 2u> p3 = {1, 1}; 
    Vec<Float, 2u> p4 = {0, 0}; 

    Vec<Float, 2u> p5 = {1, -2.5}; 
    Vec<Float, 2u> p6 = {3, 0}; 
    Vec<Float, 2u> p7 = {-1, 1}; 
    Vec<Float, 2u> p8 = {1, 2.75}; 

    Vec<Vec<Float, 2u>, 4u> curveOneData = {p1, p2, p3, p4}; 
    Vec<Vec<Float, 2u>, 4u> curveTwoData = {p5, p6, p7, p8}; 

    dray::bezier_clipping::Curve<3> curve;
    dray::bezier_clipping::Curve<3> curveTwo;
    
    int32 i = 0;
    for (auto &coeff : curve.components())
        coeff = curveOneData[i++]; 

    i = 0; 
    for (auto &coeff : curveTwo.components()) 
        coeff = curveTwoData[i++];
    
    Array<Float> res; 
    bool foundIntersection = dray::bezier_clipping::intersect(res, curve, curveTwo, false); 

    Float* resPtr = res.get_host_ptr(); 
    ASSERT_TRUE(foundIntersection); 
    ASSERT_EQ(res.size(), 1);
    // cout << "THE RESULT IS: " << resPtr[0] << endl;

    // Intersection occurs at (x, y) where (0.86 < x < 0.87) and (0.61 < y < 0.62)
}

// ===================================
// === Ray-Mesh Intersection Tests === 
// ===================================


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
