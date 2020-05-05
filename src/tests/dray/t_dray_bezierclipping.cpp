// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/vec.hpp>
#include <dray/types.hpp>
#include <dray/array.hpp>
#include <dray/range.hpp>
#include <dray/GridFunction/bezier_clipping.hpp>

using namespace dray; 

using Vec2D = bezier_clipping::Vec2D;
using Vec3D = bezier_clipping::Vec3D;

template <int p>
bezier_clipping::Curve<p - 1> create_curve(Vec<Vec2D, p> points) {
    bezier_clipping::Curve<p - 1> curve;
    uint32 i = 0;
    for (auto &point : curve.components())
        point = points[i++];
    
    return curve;
}

// ======================================
// === Normalized Implicit Line Tests ===
// ======================================
TEST(dray_bezier, dray_bezier_normalized_implicit) {
    Vec<Vec2D, 3u> points = {Vec2D{0, 0}, Vec2D{10, 5}, Vec2D{1, 1}};
    bezier_clipping::Curve<2> curve = create_curve(points);
    bezier_clipping::FatLine line = bezier_clipping::normalized_implicit(curve); 
    
    // We should have a line from (0, 0) to (1, 1).
    Vec2D control_point_one = {0, 0};
    Vec2D control_point_two = {1, 1}; 
    Float orthogonal_dist   = pow((pow(0.5, 2) + pow(0.5, 2)), 0.5);

    // Verify that the line is correct
    ASSERT_EQ(line.P_0, control_point_one);
    ASSERT_FLOAT_EQ(dot(line.N, control_point_two - control_point_one), 0.);

    // Verify that the distances are correct 
    ASSERT_FLOAT_EQ(line.dist(control_point_one), 0.);
    ASSERT_FLOAT_EQ(line.dist(control_point_two), 0.);
    ASSERT_FLOAT_EQ(line.dist(Vec2D{0, 1}), orthogonal_dist); 
}

// ======================
// === Fat Line Tests ===
// ======================
TEST (dray_bezier, dray_bezier_fatline_deg_one) {
    Vec<Vec2D, 2u> points = {Vec2D{0, 0}, Vec2D{1, 0}};

    bezier_clipping::Curve<1> curve = create_curve(points); 
    bezier_clipping::FatLine l = {Vec2D{0, 1}, Vec2D{0, 0}};

    // If we only have two points, the fat line will always have a bound of 0.
    bezier_clipping::fat_line_bounds(l, curve);
    ASSERT_EQ(l.bound, Range::zero()); 
}

TEST (dray_bezier, dray_bezier_fatline_deg_two) {
    Vec<Vec2D, 3u> points = {Vec2D{0, 0}, Vec2D{0.5, -10}, Vec2D{1, 0}};

    bezier_clipping::Curve<2> curve = create_curve(points); 
    bezier_clipping::FatLine l = {Vec2D{0, -1}, Vec2D{0, 0}};

    // In the special case of deg = 2, we take the bounds to be half of the signed distances. 
    // Dist is not -5 instead of positive 5 because the normal vector points downwards.
    bezier_clipping::fat_line_bounds(l, curve);
    ASSERT_EQ(l.bound, Range::new_range(0, 5)); 
} 

TEST (dray_bezier, dray_bezier_fatline_deg_three) {
    Vec<Vec2D, 4u> points = {Vec2D{0, 0}, Vec2D{3, 9}, Vec2D{6, -18}, Vec2D{10, 0}};
    
    bezier_clipping::Curve<3> curve = create_curve(points); 
    bezier_clipping::FatLine l = {Vec2D{0, 1}, Vec2D{0, 0}};
    bezier_clipping::fat_line_bounds(l, curve);
    ASSERT_EQ(l.bound, Range::new_range(-8, 4)); 
}

TEST (dray_bezier, dray_bezier_fatline_deg_four) {
    Vec<Vec2D, 5u> points = {Vec2D{0, 0}, Vec2D{2.5, 10}, Vec2D{5, 12}, Vec2D{7, -6}, Vec2D{10, 0}};
    
    bezier_clipping::Curve<4> curve = create_curve(points); 
    bezier_clipping::FatLine l = {Vec2D{0, 1}, Vec2D{0, 0}};
    bezier_clipping::fat_line_bounds(l, curve);
    ASSERT_EQ(l.bound, Range::new_range(-6, 12));
}

// =================================
// === Intersection Points Tests === 
// =================================
TEST(dray_bezier, dray_bezier_intersection_points_no_intersections) {
    Vec<Vec2D, 3u> points = {Vec2D{0, 10}, Vec2D{1, 5}, Vec2D{2, 10}};
    
    bezier_clipping::Curve<2> curve = create_curve(points); 
    bezier_clipping::FatLine fat_line = bezier_clipping::FatLine{Vec2D{0, 1}, Vec2D{0, 0},
                                                                 Range::new_range(-1, 1)};

    // Takes a curve and a fatline and finds the two points of intersection (if they exist). 
    // Here our fat line goes along (0, 0) -> (1, 0).
    Range t_range;
    ASSERT_FALSE(bezier_clipping::intersection_points(curve, fat_line, t_range));  
}

TEST(dray_bezier, dray_bezier_intersection_points_two_points) { 
    Vec<Vec2D, 3u> points = {Vec2D{0, 20}, Vec2D{1, 0}, Vec2D{2, -20}};
    
    bezier_clipping::Curve<2> curve = create_curve(points); 

    // The points are equally spaced out, which means we have 
    // (0, 20), (0.5, 0), and (1, -20). 
    // This gives us a slope of -40 throughout. 
    bezier_clipping::FatLine fat_line = bezier_clipping::FatLine{Vec2D{0, 1}, Vec2D{0, 0}, 
                                                                Range::new_range(-10, 10)};

    Range t_range;
    ASSERT_TRUE(bezier_clipping::intersection_points(curve, fat_line, t_range));  
    ASSERT_FLOAT_EQ(t_range.min(), 0.25); 
    ASSERT_FLOAT_EQ(t_range.max(), 0.75);
}

TEST(dray_bezier, dray_bezier_intersection_points_two_points_under) { 
    Vec<Vec2D, 3u> points = {Vec2D{0, 20}, Vec2D{1, 0}, Vec2D{2, 20}};
    
    bezier_clipping::Curve<2> curve = create_curve(points);
    // The points are equally spaced out, which means we have 
    // (0, 20), (0.5, 0), and (1, -20). 
    // This gives us a slope of -40 throughout.  
    bezier_clipping::FatLine fatLine = bezier_clipping::FatLine{Vec2D{0, 1}, Vec2D{0, 0},
                                                                Range::new_range(-10, 10)};

    Range t_range;
    ASSERT_TRUE(bezier_clipping::intersection_points(curve, fatLine, t_range));  
    ASSERT_FLOAT_EQ(t_range.min(), 0.25); 
    ASSERT_FLOAT_EQ(t_range.max(), 0.75);
}

// ==========================
// === Intersection Tests === 
// ==========================
TEST(dray_bezier, dray_bezier_intersect_one_intersection) { 
    Vec<Vec2D, 2u> points = {Vec2D{0, 0}, Vec2D{1, 0}};
    Vec<Vec2D, 2u> points_two = {Vec2D{0.5, 0.5}, Vec2D{0.5, -0.5}};

    bezier_clipping::Curve<1> curve = create_curve(points);
    bezier_clipping::Curve<1> curve_two = create_curve(points_two);

    Array<Float> res, res_two;

    ASSERT_EQ(bezier_clipping::intersect(res, curve, curve_two, 1), 1);
    Float* res_ptr = res.get_host_ptr(); 
    ASSERT_EQ(res.size(), 1); 
    ASSERT_FLOAT_EQ(res_ptr[0], 0.5); 

    cout << "Checking the second curve against the first" << endl; 
    ASSERT_EQ(bezier_clipping::intersect(res_two, curve_two, curve, 1), 1); 
    ASSERT_EQ(res_two.size(), 1); 
    ASSERT_FLOAT_EQ(res_two.get_host_ptr()[0], 0.5);  
}

TEST(dray_bezier, dray_bezier_intersect_no_intersection) { 
    Vec<Vec2D, 2u> points     = {Vec2D{0, 0}, Vec2D{1, 0}};
    Vec<Vec2D, 2u> points_two = {Vec2D{0, 0.5}, Vec2D{1, 0.5}};

    bezier_clipping::Curve<1> curve = create_curve(points);
    bezier_clipping::Curve<1> curve_two = create_curve(points_two);
    
    Array<Float> res;
    ASSERT_EQ(bezier_clipping::intersect(res, curve, curve_two), 0);
    ASSERT_EQ(res.size(), 0); 
} 

TEST(dray_bezier, dray_bezier_multiple_iterations) { 
    // Demo: https://www.desmos.com/calculator/dqt4io0faw
    Vec<Vec2D, 4u> curve_one_data = {Vec2D{3, 0}, Vec2D{2, 1}, Vec2D{1, 1}, Vec2D{0, 0}}; 
    Vec<Vec2D, 4u> curve_two_data = {Vec2D{1, -2.5}, Vec2D{3, 0}, Vec2D{-1, 1}, Vec2D{1, 2.75}};  

    bezier_clipping::Curve<3> curve = create_curve(curve_one_data);
    bezier_clipping::Curve<3> curve_two = create_curve(curve_two_data);
    
    Array<Float> res; 
    ASSERT_EQ(bezier_clipping::intersect(res, curve, curve_two, 20), 1); 
    Float* res_ptr = res.get_host_ptr(); 
    ASSERT_EQ(res.size(), 1);
    // Intersection occurs at (x, y) where (0.86 < x < 0.87) and (0.61 < y < 0.62)
}
