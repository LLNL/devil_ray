#include <dray/GridFunction/bezier_clipping.hpp>
#include <dray/Element/bernstein_basis.hpp>
#include <stack> 

namespace dray 
{

namespace bezier_clipping 
{

// ============================
// === Curve-Curve Clipping === 
// ============================

typedef Vec<Float, 3u> Vec3D; 
typedef Vec<Float, 2u> Point;

template <uint32 p_order>
using Curve = MultiVec<Float, 1, 2, p_order>;

template <uint32 p_order> 
bool intersection_points(Curve<p_order> curve, FatLine line, Array<Float> &intersections) {
    // At most two intersections are possible. 
    intersections.resize(2); 
    Float* intersectionsPtr = intersections.get_host_ptr();    
    bool foundIntersection = false; 

    // Assume that the control points are in order.
    size_t numControlPoints = curve.size(); 

    Point prevPoint         = curve[0]; 
    Point currPoint; 


    // Compute the distance between each control point in the 't' axis. 
    Float t_interval        = (1. / (numControlPoints - 1));
    Float t_0 = 0;   // t value for previous control point
    Float t_1 = 0;   // t value for current control point
    Float t_min = std::numeric_limits<Float>::infinity(); // minimum t value for intersection range 
    Float t_max = -1 * std::numeric_limits<Float>::infinity(); // maximum t value for intersection range

    // Iterate through the control points. 
    int i = 0;
    for (auto &currPoint : curve.components()) {
        if (i == 0) {
            i++;
            continue;
        }
        
    // for (size_t i = 1; i < numControlPoints; ++i) {
        // Find the intersection between the line from current control point to previous control point
        // and the lines bounding the fat line.
        bool intersect = t_intersection(line, prevPoint, currPoint, t_min, t_max, t_0, t_1, t_interval);
        foundIntersection = foundIntersection || intersect; 
    }

    // Consider the line segment from P_0 to last control point. 
    t_0 = 0; 
    t_1 = 0; 
    t_interval = Float(1); 
    bool intersect = t_intersection(line, 
                                    curve[0], 
                                    curve[numControlPoints - 1], 
                                    t_min, t_max, t_0, t_1, 
                                    t_interval);
    foundIntersection = foundIntersection || intersect; 
    
    if (foundIntersection) {           
        intersectionsPtr[0] = t_min; 
        intersectionsPtr[1] = t_max; 
    }

    return foundIntersection; 
}

template <uint32 p_order1, uint32 p_order2>
bool intersect(Array<Float> &res, 
               Curve<p_order1> &curveOne,
               Curve<p_order2> &curveTwo,
               int maxIterations = 10,
               float threshold = 1e-3) {
    NormalizedImplicitLine line;
    FatLine fatLine; 
    Array<Float> intersections;
    Float t_min, t_max; 
    Float* resPtr = res.get_host_ptr();

    stack<Curve<p_order1>> paramStack;
    stack<Curve<>> paramStack2;
    Curve<p_order1> clipCurve  = curveOne; 
    Curve<p_order2> otherCurve = curveTwo;

    size_t num_intersections = 0; 

    while (true) {
        if (maxIterations == 0)
            return false;
        --maxIterations;

        line    = normalized_implicit(clipCurve);

        // Point* temp = clipCurve.get_host_ptr(); 
        fatLine = fat_line(line, clipCurve);
        
        // Find the points of intersection with between the second curve and 
        // the fat line. 
    
        bool foundIntersections = intersection_points(otherCurve, fatLine, intersections); 
        Float* pointsPtr = intersections.get_host_ptr(); 
        if (!foundIntersections) {
            if (paramStack.size() == 0) 
                break; 

            clipCurve = paramStack.top();
            paramStack.pop(); 
            otherCurve = paramStack.top(); 
            paramStack.pop(); 
            continue;
        }
        
        t_min = pointsPtr[0];
        t_max = pointsPtr[1]; 

        // Base case
        if ((t_max - t_min) < threshold) {
            res.resize(res.size() + 1);

            resPtr = res.get_host_ptr();    
            resPtr[res.size() - 1] = (t_max + t_min) / 2; 

            if (paramStack.size() == 0) 
                ++num_intersections;
                break; 
            
            clipCurve  = paramStack.top(); 
            paramStack.pop();
            otherCurve = paramStack.top(); 
            paramStack.pop(); 
            continue; 
        }

        // If we managed to clip less than 20% of the interval, there are probably two intersections. Split the curve into two 
        // and recursively call intersect() on the two halves.
        if ((t_max - t_min) > 0.80) {
            Curve<p_order1> segmentOne; 

            // Copy the curve to segmentOne.
            int i = 0;
            for (auto &pt : segmentOne.components()) 
                pt = otherCurve[i++];

            // Make segment one the first half of the curve.
            dray::DeCasteljau::split_inplace_left<Curve<p_order1>>(segmentOne, (Float) t_max, (uint32) otherCurve.size());

            // Make otherCurve the second half of the curve.
            dray::DeCasteljau::split_inplace_right<Curve<p_order1>>(otherCurve, (Float) t_min, (uint32) otherCurve.size());

            paramStack.push(otherCurve); 
            paramStack.push(clipCurve);  
            otherCurve = segmentOne;
            continue; 
        }  

        // Split at t_max
        dray::DeCasteljau::split_inplace_left<Curve<p_order1>>(otherCurve, (Float) t_max, (uint32) otherCurve.size());
        
        // Split at t_min
        // TODO: shouldn't the new t_min be different after the split>
        dray::DeCasteljau::split_inplace_right<Curve<p_order1>>(otherCurve, (Float) t_min, (uint32) otherCurve.size());

        Curve<p_order1> temp = otherCurve; 
        otherCurve = clipCurve;
        clipCurve  = otherCurve; 
    }

    return num_intersections > 0; 
}

template <uint32 p_order>
FatLine fat_line(NormalizedImplicitLine l, Curve<p_order> curveOne) {
    size_t degree   = curveOne.size() - 1;  
    
    Float minDist = l.dist(curveOne[0]);  
    Float maxDist = l.dist(curveOne[0]);
   
    // We can use tight bounds in the case where the curve is quadratic
    if (degree == 2) { 
        Point p = curveOne[1]; 
        Float scaled_p_dist = l.dist(p) / 2;
        minDist = std::min(Float(0), scaled_p_dist);
        maxDist = std::max(Float(0), scaled_p_dist);  
    } else if (degree == 3) { 
        // We can user tigher bounds in the case where the curve is cubic 
        Point p1 = curveOne[1];
        Point p2 = curveOne[2];
        Float p1_dist = l.dist(p1); // -18
        Float p2_dist = l.dist(p2); // 9
        minDist = (4. / 9) * std::min(std::min(p1_dist, p2_dist), Float(0));
        maxDist = (4. / 9) * std::max(std::max(p1_dist, p2_dist), Float(0));
    } else {
        for (size_t i = 1; i < degree + 1; ++i) {
            Point p = curveOne[i];
            Float dist = l.dist(p); 
            if (dist < minDist) {
                minDist = dist;
            }
            if (dist > maxDist) {
                maxDist = dist;
            }
        }
    }

    FatLine fatLine        = FatLine{};
    fatLine.upperBound     = maxDist;
    fatLine.lowerBound     = minDist;
    fatLine.line           = l; 
    return fatLine;
}

bool t_intersection(FatLine& line,
                    Point& prevPoint,
                    Point& currPoint, 
                    Float& t_min, 
                    Float& t_max, 
                    Float& t_0, 
                    Float& t_1, 
                    Float& t_interval) { 

    Float y_0  = line.line.dist(prevPoint); 
    Float y_1  = line.line.dist(currPoint); 
    t_1 += t_interval;  

    // Create a line segment by looking from the (i)th point to the (i-1)th point.
    // This starts at 'prevPoint' and goes to the currPoint. 
    Float slope = (y_1 - y_0) / (t_1 - t_0); 
    Float delta_t; 

    bool foundIntersection = false; 
    bool condOne = (y_0 <= line.upperBound && line.upperBound <= y_1);
    bool condTwo = (y_1 <= line.upperBound && line.upperBound <= y_0);
    
    if (condOne || condTwo) { 
        delta_t = (line.upperBound - y_0) / slope; 
        t_min   = std::min(t_min, t_0 + delta_t);
        t_max   = std::max(t_max, t_0 + delta_t);
        foundIntersection = true; 
    } 
    condOne = (y_0 <= line.lowerBound && line.lowerBound <= y_1);
    condTwo = (y_1 <= line.lowerBound && line.lowerBound <= y_0);

    if (condOne || condTwo) { 
        delta_t = (line.lowerBound - y_0) / slope;  
        t_min   = std::min(t_min, t_0 + delta_t);
        t_max   = std::max(t_max, t_0 + delta_t);
        foundIntersection = true; 
    } 

    // Update for next iteration
    prevPoint = currPoint; 
    t_0       = t_1;
    return foundIntersection; 
}

// Takes a bezier curve and creates a normalized implicit line through it.
template <uint32 p_order>
NormalizedImplicitLine normalized_implicit(Curve<p_order> curve) {
    // First, create a line by looking at the first and last control point
    Point firstControlPoint = curve[0]; 
    size_t size = curve.size();
    Point lastControlPoint = curve[size - 1];

    Point difference = lastControlPoint - firstControlPoint;
    
    // Get the orthogonal vector in 2D 
    Point normalVect = {-1 * difference[1], difference[0]}; 
    normalVect.normalize();

    NormalizedImplicitLine line = NormalizedImplicitLine{};
    line.N = normalVect;
    line.P_0 = firstControlPoint;
    return line; 
}

// ================================
// === Ray-Surface Intersection ===
// ================================

/* 
    We represent the ray as an intersection of two planes.
*/
void rayToIntersectingPlanes(Vec3D &rayDirection, Vec3D &planeOneNormalVec, Vec3D &planeTwoNormalVec) {
    planeOneNormalVec = {-rayDirection[1], rayDirection[0], 0};
    planeTwoNormalVec = {0, -rayDirection[2], rayDirection[1]};
}

/* 
    We project the mesh two dimensions. This new surface is also a Bezier surface, so we just 
    compute the new set of control points.

    Assume that we have control points p_ij where 
    0 <= i < n, 
    0 <= j < m. 
    Then p_ij = controlPoints[i * n + j]

    Note that we are not using rational Bezier curves, so the weights are all 1. 
*/ 
void projectTo2D(
    Vec3D &planeOneNormal,
    Vec3D &planeTwoNormal, 
    Vec3D &rayOrigin, 
    Array<Vec3D> &controlPoints,
    Array<Point> &newControlPoints,
    size_t n,
    size_t m) {
    /* 
    Adjust the size of the results array to have 
    as many points as the current number of control 
    points.
    */ 
    newControlPoints.resize(controlPoints.size());
    Point *resPtr = newControlPoints.get_host_ptr(); 
    Vec3D *controlPointsPtr = controlPoints.get_host_ptr(); 

    Float h_ij1 = 0;
    Float h_ij2 = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            size_t index = i * m + j;
            h_ij1 = dray::dot(planeOneNormal, controlPointsPtr[index]) - 
                    dray::dot(planeOneNormal, rayOrigin);
            h_ij2 = dray::dot(planeTwoNormal, controlPointsPtr[index]) - 
                     dray::dot(planeTwoNormal, rayOrigin);
            resPtr[i * n + j] = {h_ij1, h_ij2}; 
        }
    }
}

/*
 * Generate two vectors from the control points of the projected plane.
 * This is similar to creating two fat lines with d_min = d_max = 0 in the 
 * original Bezier Clipping algorithm.
 * 
 * (Step 3 of the algorithm described in the NURBs paper)
 */
void getFatLines(
        Array<Point> &newControlPoints,
        Point &Lu,
        Point &Lv,
        size_t n,
        size_t m) 
{
    Point* controlPointsPtr = newControlPoints.get_host_ptr();
    const Point r_00 = controlPointsPtr[0];
    const Point r_n0 = controlPointsPtr[n * m];
    const Point r_0m = controlPointsPtr[m - 1];
    const Point r_nm = controlPointsPtr[(n * m) + (m - 1)];

    Lu = (r_n0 - r_00) + (r_nm - r_0m);
    Lv = (r_0m - r_00) + (r_nm - r_n0);
}

bool intersectMesh(size_t n, size_t m,
                   Vec3D &rayDirection, Vec3D &rayOrigin, 
                   Array<Vec3D> controlPoints, 
                   size_t maxIterations = 10) {
    Vec3D planeOneNormal;
    Vec3D planeTwoNormal;
    rayToIntersectingPlanes(
        rayDirection,
        planeOneNormal, 
        planeTwoNormal);

    Array<Point> newControlPoints; 
    projectTo2D(planeOneNormal, planeTwoNormal, rayOrigin,
                controlPoints, newControlPoints,n, m);
   
    Point Lu;
    Point Lv;
    getFatLines(newControlPoints, Lu, Lv, n, m); 
    while (true) {
        if (--maxIterations < 0) 
            break;
    }
}

// ==============================
// === Explicit instantiation ===
// ==============================
template bool intersect(
            Array<Float> &res, 
            Curve<3> &curveOne,
            Curve<3> &curveTwo,
            int maxIterations,
            float threshold);
template bool intersect(
            Array<Float> &res, 
            Curve<1> &curveOne,
            Curve<1> &curveTwo,
            int maxIterations,
            float threshold);

// Try templating on two different orders. 
// TODO: remove
template bool intersect(
            Array<Float> &res, 
            Curve<2> &curveOne,
            Curve<1> &curveTwo,
            int maxIterations,
            float threshold);

template NormalizedImplicitLine normalized_implicit(Curve<2> curve);

template FatLine fat_line(NormalizedImplicitLine l, Curve<1> curveOne);
template FatLine fat_line(NormalizedImplicitLine l, Curve<2> curveOne);
template FatLine fat_line(NormalizedImplicitLine l, Curve<4> curveOne);

template bool intersection_points(Curve<2> curve, 
                                  FatLine line, 
                                  Array<Float> &intersections);
    

}
}
