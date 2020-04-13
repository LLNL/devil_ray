#ifndef DRAY_BEZIER_CLIPPING_HPP
#define DRAY_BEZIER_CLIPPING_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>

/*
    This implementation of the curve-curve interesection is
     based on the paper   
    "Curve Intersection using Bezier Clipping"
    by T W Sederberg and T Nishita 

    The implementation of ray-surface intersection uses the 
    "Robust and Numerically Stable Bezier Clipping Method for 
    Ray Tracing NURBS Surfaces" 
    by Alexander Efremov et al.
*/ 

namespace dray 
{

namespace bezier_clipping 
{

typedef Vec<Float, 2u> Vec2D;
typedef Vec<Float, 3u> Vec3D;

template <uint32 p_order>
using Curve = MultiVec<Float, 1, 2, p_order>;

// ============================================  
// BEZIER CLIPPING FOR CURVE-CURVE INTERSECTION
// ============================================

// A NormalizedImplicitLine is one of the form 
// <N, (x - p_0)> = 0
// i.e. the dot product of N and (x - p_o), where 
// p_o is one of the points on the line. Further,
// the norm of N equals 1: ||N|| = 1. 

// Computes the distance to an arbitrary point. 
struct NormalizedImplicitLine {
    Float dist(Vec2D point) {
        // If the line is of the form 
        // N*X - N*P_0 = 0 
        // Then the distance from an arbitrary point (x, y) 
        // is d(x, y) = ax + by + c
        return dot(N, point) - dot(N, P_0);   
    }

    Vec2D N;
    Vec2D P_0;

    friend std::ostream& operator<<(std::ostream& out, const NormalizedImplicitLine &line) {
        return out << "N: " << line.N << " P_0: " << line.P_0;
    } 
};

// A FatLine is a NoramlizedImplicitLine with bounds around it. 
struct FatLine {
    Float upper_bound;
    Float lower_bound;  

    NormalizedImplicitLine line;

    friend std::ostream& operator<<(std::ostream& out, const FatLine &line) {
        return out << "Upper Bound: " << line.upper_bound << " Lower Bound: " << line.lower_bound;
    } 
};

// TODO: temp
template <uint32 p_order> 
void de_casteljau(Curve<p_order> &resOne,
                  Curve<p_order> &resTwo,
                  Curve<p_order> input,
                  Vec2D intersectPoint);

bool t_intersection(FatLine& line,
                    Vec2D& prevPoint,
                    Vec2D& currPoint,
                    Float& t_min,
                    Float& t_max, 
                    Float& t_0, 
                    Float& t_1, 
                    Float& t_interval);

// Takes a bezier curve and creates a normalized implicit line through it.
template <uint32 p_order>
NormalizedImplicitLine normalized_implicit(Curve<p_order> curve);
NormalizedImplicitLine to_normalized_implicit(Vec2D originalLine, Vec2D p_0);

template <uint32 p_order> 
bool intersection_points(Curve<p_order> curve,
                         FatLine line, 
                         Array<Float> &intersections);

// Take two bezier curves, find their intersection
template <uint32 p_order1, uint32 p_order2>
bool intersect(Array<Float> &res,
               Curve<p_order1> &curveOne, 
               Curve<p_order2> &curveTwo, 
               int maxIterations = 10, 
               float threshold = 1e-3,
               bool returnClosestSolution = false);

// fat_line creates a FatLine from a bezier curve and a NormalizedImplicitLine.
// 
// This is done by taking the maximum (signed) distance from each control point to 
// the NormalizedImplicitLine.
template <uint32 p_order>
FatLine fat_line(NormalizedImplicitLine l, Curve<p_order> curveOne);

// ============================================  
// BEZIER CLIPPING FOR RAY-SURFACE INTERSECTION
// ============================================

/*
    Computes 
        n_1 = (a_1, b_1, c_1) = (-d_y, d_x, 0)
        n_2 = (a_2, b_2, c_2) = (0, -d_z, d_y)
*/
void getDistanceControlPoints(Array<Vec2D> &distanceControlPoints, FatLine fatLine, 
                              Array<Vec2D> controlPoints, int n, int m, bool directionU);
void getConvexHull(Array<Vec2D> convexHull, Array<Vec2D> controlPoints, int n, int m, bool directionU);
void projectTo2D(Vec3D &planeOneNormal, Vec3D &planeTwoNormal, 
    Vec3D &rayOrigin, Array<Vec3D> &controlPoints, 
    Array<Vec2D> &newControlPoints, size_t n, size_t m);
void rayToIntersectingPlanes(Vec3D rayDirection);
void h(int k, int i, int j);
void r(int i, int j); 
void d_k(int );

}
}
#endif // DRAY_BEZIER_CLIPPING_HPP
