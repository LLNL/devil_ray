#include "gtest/gtest.h"
#include <dray/vec.hpp>
#include <dray/types.hpp>
#include <dray/array.hpp>
#include <dray/GridFunction/bezier_clipping_mesh.hpp>
#include <dray/GridFunction/bezier_clipping.hpp>

using namespace dray;

using Vec3D = bezier_clipping::Vec3D;
using Vec2D = bezier_clipping::Vec2D; 

template <uint32 p_order>
bezier_clipping::Mesh3D<p_order> create_mesh(Vec<Vec3D, (p_order + 1) * (p_order + 1)> points) {
    bezier_clipping::Mesh3D<p_order> mesh; 
    int i = 0;
    for (int row_index = 0; row_index <= p_order; ++row_index) { 
        for (auto &point : mesh[row_index].components())
            point = points[i++];
    }

    return mesh;
}

template <uint32 p_order> 
bezier_clipping::Mesh2D<p_order> create_mesh_2d(Vec<Vec2D, (p_order + 1) * (p_order + 1)> points) {
    bezier_clipping::Mesh2D<p_order> mesh; 
    int i = 0; 
    for (int row_index = 0; row_index <= p_order; ++row_index) {
        for (auto &point : mesh[row_index].components())
            point = points[i++];
    }

    return mesh;
}

// ===================================
// === Ray-Mesh Intersection Tests === 
// ===================================

TEST(dray_bezier, dray_bezier_ray_to_planes) {
    Vec3D vec1 = Vec3D{1., 0., 0.}; 
    Vec3D vec2 = Vec3D{0., 1., 0.}; 
    Vec3D vec3 = Vec3D{0., 0., 1.}; 
    Vec3D vec4 = Vec3D{1., 1., 0.}; 
    Vec3D vec5 = Vec3D{0., 1., 1.}; 
    Vec3D vec6 = Vec3D{1., 0., 1.}; 
    Vec3D vec7 = Vec3D{1., 1., 1.}; 
    Vec<Vec3D, 7> vecs = {vec1, vec2, vec3, vec4, vec5, vec6, vec7};

    for (int i = 0; i < 7; ++i) {
        vecs[i].normalize(); 

        bezier_clipping::Plane p1, p2;
        dray::Ray ray;
        ray.m_orig = Vec3D{0., 0., 0.};
        ray.m_dir = vecs[i];   

        bezier_clipping::ray_to_planes(p1, p2, ray);    
        Vec3D res = cross(p1.n, p2.n);
        EXPECT_NEAR(res[0], vecs[i][0], 0.00001);
        EXPECT_NEAR(res[1], vecs[i][1], 0.00001);
        EXPECT_NEAR(res[2], vecs[i][2], 0.00001); 
        EXPECT_FLOAT_EQ(p1.o, 0);
        EXPECT_FLOAT_EQ(p2.o, 0); 
    }

    dray::Ray ray; 
    ray.m_dir = Vec3D{1., 1., 1.};
    ray.m_orig = Vec3D{1., 2., 3.};
    bezier_clipping::Plane p1, p2;
    bezier_clipping::ray_to_planes(p1, p2, ray);    
    ASSERT_FLOAT_EQ(dot(p1.n, Vec3D{1., 2., 3.}) + p1.o, 0);
    ASSERT_FLOAT_EQ(dot(p2.n, Vec3D{1., 2., 3.}) + p2.o, 0);

    ASSERT_NEAR(dot(p1.n, Vec3D{4., 5., 6.}) + p1.o, 0, 0.00001);
    ASSERT_NEAR(dot(p2.n, Vec3D{4., 5., 6.}) + p2.o, 0, 0.00001);
}

TEST(dray_bezier, dray_bezier_project_mesh) {
    Vec<Vec3D, 9u> points = {Vec3D{1, 0, 0}, Vec3D{0, 1, 0}, Vec3D{0, 0, 1},
                             Vec3D{1, 1, 1}, Vec3D{-1, -1, -1}, Vec3D{2, 2, 2},
                             Vec3D{0, 0, 0}, Vec3D{0, 0, 0}, Vec3D{0, 0, 0}};

    const bezier_clipping::Plane p1 = bezier_clipping::Plane{Vec3D{1, 2, 3}, 0.};
    const bezier_clipping::Plane p2 = bezier_clipping::Plane{Vec3D{0, 0, 0}, 1.};
    bezier_clipping::Mesh3D<2u> mesh = create_mesh<2u>(points);
    bezier_clipping::Mesh2D<2u> projected_mesh = bezier_clipping::project_mesh<2u>(p1, p2, mesh); 
    
    Vec<Float, 9u> expected = {1, 2, 3,
                               6, -6, 12,
                               0, 0, 0};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_FLOAT_EQ(projected_mesh[i][j][1], 1.);
            ASSERT_FLOAT_EQ(projected_mesh[i][j][0], expected[i*3 + j]);
        }
    }
}

TEST(dray_bezier, dray_bezier_parametrized_directions) {
    bezier_clipping::FatLine l_u, l_v;
    Vec<Vec2D, 9u> points = {Vec2D{0, 5}, Vec2D{8, 2}, Vec2D{3, 2},
                             Vec2D{4, 3}, Vec2D{1, 5}, Vec2D{3, 6},
                             Vec2D{2, 7}, Vec2D{0, 4}, Vec2D{5, 2}};
    bezier_clipping::Mesh2D<2u> mesh = create_mesh_2d<2u>(points); 
    bezier_clipping::parametrized_directions(l_u, l_v, mesh);   

    // ((2, 7) - (0, 5)) + ((5, 2) - (3, 2)) 
    // = (2, 2) + (2, 0) = (4, 2)

    // ((3, 2) - (0, 5)) + ((5, 2) - (2, 7)) 
    // = (3, -3) + (3, -5) = (6, -8)

    Vec2D expected_one = Vec2D{4, 2};
    Vec2D expected_two = Vec2D{6, -8};
    ASSERT_EQ(dot(l_u.N, expected_one), 0);
    ASSERT_EQ(dot(l_v.N, expected_two), 0);
}

TEST(dray_bezier, dray_bezier_explicit_mesh) {
    bezier_clipping::FatLine l;
    l.N = Vec2D{0., 1.};
    l.P_0 = Vec2D{0., 0.};
    l.bound = Range::zero();
    Vec<Vec2D, 9u> points = {Vec2D{0, 0}, Vec2D{0, 0}, Vec2D{0, 0},
                                              Vec2D{0, 0}, Vec2D{0, 0}, Vec2D{0, 0},
                                              Vec2D{0, 0}, Vec2D{0, 0}, Vec2D{0, 0}};
    bezier_clipping::Mesh2D<2u> projected_mesh = create_mesh_2d<2u>(points); 
    bezier_clipping::Mesh2D<2u> res = bezier_clipping::create_explicit_mesh<2u>(l, projected_mesh, true);
    
    Vec<Vec2D, 9u> points2 = {Vec2D{0, 0},   Vec2D{0, 0},   Vec2D{0, 0},
                              Vec2D{0.5, 0}, Vec2D{0.5, 0}, Vec2D{0.5, 0},
                              Vec2D{1, 0},   Vec2D{1, 0},   Vec2D{1, 0}};
    bezier_clipping::Mesh2D<2u> expected = create_mesh_2d<2u>(points2);
    for (int i = 0; i < 3; ++i) { 
        for (int j = 0; j < 3; ++j) {
            ASSERT_FLOAT_EQ(expected[i][j][0], res[i][j][0]);
            ASSERT_FLOAT_EQ(expected[i][j][1], res[i][j][1]);
        }
    }

    points2 = {Vec2D{0, 0}, Vec2D{0.5, 0}, Vec2D{1, 0},
               Vec2D{0, 0}, Vec2D{0.5, 0}, Vec2D{1, 0},
               Vec2D{0, 0}, Vec2D{0.5, 0}, Vec2D{1, 0}};
    expected = create_mesh_2d<2u>(points2);
    bezier_clipping::Mesh2D<2u> res2 = bezier_clipping::create_explicit_mesh<2u>(l, projected_mesh, false);
    for (int i = 0; i < 3; ++i) { 
        for (int j = 0; j < 3; ++j) {
            ASSERT_FLOAT_EQ(expected[i][j][0], res2[i][j][0]);
            ASSERT_FLOAT_EQ(expected[i][j][1], res[i][j][1]);
        }
    }
    // res = bezier_clipping::explicit_mesh<2u>(l, projected_mesh, false);
    // cout << res << endl;
}

TEST(dray_bezier, dray_bezier_region_of_interest) {
    bool u_direction = true;
    Range t_range;
    Vec<bezier_clipping::Vec2D, 9u> points = {Vec2D{0, -1},   Vec2D{0, -2},   Vec2D{0, -3},
                                              Vec2D{0.5, -1}, Vec2D{0.5, 1}, Vec2D{0.5, 2},
                                              Vec2D{1, 1},    Vec2D{1, 2},   Vec2D{1, 3}};
    bezier_clipping::Mesh2D<2u> explicit_mesh = create_mesh_2d<2u>(points); 
    bezier_clipping::region_of_interest<2u>(explicit_mesh, t_range, u_direction);    
    ASSERT_FLOAT_EQ(t_range.min(), 0.16666666667);
    ASSERT_FLOAT_EQ(t_range.max(), 0.75);
}

TEST(dray_bezier, dray_bezier_region_of_interest_edge_case) {
    bool u_direction = false;
    Range t_range;
    Vec<bezier_clipping::Vec2D, 9u> points = {Vec2D{0, -1}, Vec2D{0.5, 1}, Vec2D{1, -1},
                                              Vec2D{0, 1},  Vec2D{0.5, 2}, Vec2D{1, 2},
                                              Vec2D{0, 2},  Vec2D{0.5, 3}, Vec2D{1, 3}};
    bezier_clipping::Mesh2D<2u> explicit_mesh = create_mesh_2d<2u>(points); 
    bezier_clipping::region_of_interest<2u>(explicit_mesh, t_range, u_direction);    
    ASSERT_FLOAT_EQ(t_range.min(), 0);
    ASSERT_FLOAT_EQ(t_range.max(), 1);
}

TEST(dray_bezier, dray_bezier_intersect) { 
    Vec3D intersection;
    Ray ray;
    ray.m_dir  = Vec3D{0, 0, -1};
    ray.m_orig = Vec3D{0, 0, 1};
    Vec<Vec3D, 4u> points = {Vec3D{-1, 1, 0}, Vec3D{1, 1, 0},
                             Vec3D{-1, -1, 1}, Vec3D{1, -1, -1}};
    bezier_clipping::Mesh3D<1u> mesh = create_mesh<1u>(points); 
    Array<Float> res;
    int num_intersections = bezier_clipping::intersect_mesh(res, ray, mesh, intersection, 10, .001);
    cout << num_intersections << endl; 
}