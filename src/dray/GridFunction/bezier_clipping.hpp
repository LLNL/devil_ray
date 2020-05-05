#ifndef DRAY_BEZIER_CLIPPING_HPP
#define DRAY_BEZIER_CLIPPING_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/range.hpp>

/*
    This implementation of the curve-curve interesection is based on the paper 
    "Curve Intersection using Bezier Clipping" by T W Sederberg and T Nishita.

    The implementation of ray-surface intersection uses the "Robust and Numerically
    Stable Bezier Clipping Method for Ray Tracing NURBS Surfaces" by Alexander Efremov et al.
*/ 

namespace dray 
{

namespace bezier_clipping 
{

// ============================================  
// BEZIER CLIPPING FOR CURVE-CURVE INTERSECTION
// ============================================
typedef Vec<Float, 2u> Vec2D;
typedef Vec<Float, 3u> Vec3D;

template <uint32 p_order>
using Curve = MultiVec<Float, 1, 2, p_order>;

template <uint32 p>
Curve<p> create_curve(Curve<p> points) {
    bezier_clipping::Curve<p> curve;
    uint32 i = 0;
    for (auto &point : curve.components())
        point = points[i++];
    
    return curve;
}

// A FatLine is a NoramlizedImplicitLine with bounds around it. 
// A NormalizedImplicitLine is one of the form "<N, (x - p_0)> = 0"
// i.e. the dot product of N and (x - p_o), where p_o is one of the points on the line.
// Further, the norm of N equals 1: ||N|| = 1. 
struct FatLine {
    Vec2D N;
    Vec2D P_0;
    Range bound;

    Float dist(Vec2D point) const {
        // If the line is of the form "N*X - N*P_0 = 0",
        // Then the distance from an arbitrary point (x, y) is d(x, y) = ax + by + c.
        return dot(N, point) - dot(N, P_0);   
    }

    friend std::ostream& operator<<(std::ostream& out, const FatLine &line) {
        return out << line.bound;
    } 
};

inline bool operator==(const FatLine& lhs, const FatLine& rhs) {
    return lhs.bound == rhs.bound;
}

bool t_intersection(FatLine& line, Float y_0, Float y_1, Range &t_range, Float t_0, Float t_1);

// Takes a bezier curve and creates a normalized implicit line through it.
template <uint32 p_order>
FatLine normalized_implicit(Curve<p_order> curve);

template <uint32 p_order> 
bool intersection_points(Curve<p_order> curve, FatLine line, Range& t_range);

// Take two bezier curves, find their intersection.
template <uint32 p_order1, uint32 p_order2>
int intersect(Array<Float> &res, Curve<p_order1> &curve_one, Curve<p_order2> &curve_two,
               int max_iterations = 10, float threshold = 1e-3);


// Take the maximum (signed) distance from each control point to the FatLine.
template <uint32 p_order>
void fat_line_bounds(FatLine& l, Curve<p_order> curve_one);

}
}
#endif // DRAY_BEZIER_CLIPPING_HPP
