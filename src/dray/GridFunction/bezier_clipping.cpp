#include <dray/GridFunction/bezier_clipping.hpp>
#include <dray/Element/bernstein_basis.hpp>
#include <dray/ray.hpp>
#include <stack> 

namespace dray 
{

namespace bezier_clipping 
{

// ============================
// === Curve-Curve Clipping === 
// ============================

typedef Vec<Float, 2u> Vec2D;
typedef Vec<Float, 3u> Vec3D; 

template <uint32 p_order>
using Curve = MultiVec<Float, 1, 2, p_order>;

template <uint32 p_order> 
bool intersection_points(Curve<p_order> curve, FatLine line, Array<Float> &intersections) {
    // At most two intersections are possible. 
    intersections.resize(2); 
    Float* intersections_ptr = intersections.get_host_ptr();    
    bool found_intersection = false; 

    // Assume that the control points are in order.
    size_t numControlPoints = curve.size(); 

    Vec2D prevPoint         = curve[0]; 
    Vec2D currPoint; 

    // Compute the distance between each control point in the 't' axis. 
    Float t_interval        = (1. / (numControlPoints - 1));
    Float t_0 = 0;   // t value for previous control point.
    Float t_1 = 0;   // t value for current control point.
    Float t_min = std::numeric_limits<Float>::infinity(); // minimum t value for intersection range.
    Float t_max = -1 * std::numeric_limits<Float>::infinity(); // maximum t value for intersection range.

    // Iterate through the control points. 
    int i = 0;
    for (auto &currPoint : curve.components()) {
        if (i == 0) {
            i++;
            continue;
        }

        // Find the intersection between the line from current control point to previous control point
        // and the lines bounding the fat line.
        bool intersect = t_intersection(line, prevPoint, currPoint, t_min, t_max, t_0, t_1, t_interval);
        found_intersection = found_intersection || intersect; 
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
    found_intersection = found_intersection || intersect; 

    if (found_intersection) {           
        intersections_ptr[0] = t_min; 
        intersections_ptr[1] = t_max; 
    }

    return found_intersection; 
}

template <uint32 p_order1, uint32 p_order2>
union CurveData {
    Curve<p_order1> curve_order_one; 
    Curve<p_order2> curve_order_two;
};

template <uint32 p_order1, uint32 p_order2> 
struct CurveStruct { 
    CurveData<p_order1, p_order2> curve_data; 
    bool orderOne;
    Float t_min;
    Float t_max;

    uint32 size() { 
        if (orderOne)
            return (uint32) curve_data.curve_order_one.size();

        return (uint32) curve_data.curve_order_two.size();
    }
};

template <uint32 p_order1, uint32 p_order2>
bool intersect(Array<Float> &res, 
               Curve<p_order1> &curve_one,
               Curve<p_order2> &curve_two,
               int maxIterations = 10,
               Float threshold = 0.25f,
               bool returnClosestSolution = false) {
    
    using CurveData   = CurveData<p_order1, p_order2>; 
    using CurveStruct = CurveStruct<p_order1, p_order2>; 
    stack<CurveStruct> param_stack;
    Array<Float> intersections;
    Float t_min, t_max; 

    // Initialize the CurveStructs.
    CurveData data_one;
    data_one.curve_order_one = curve_one; 
    CurveData data_two;
    data_two.curve_order_two = curve_two;    
    CurveStruct clip_curve = {data_one, true, 0, 1}; 
    CurveStruct other_curve = {data_two, false, 0, 1};
    size_t num_intersections = 0; 

    while (true) {
        if (--maxIterations < 0)
            return false;
        
        // Find the points of intersection with between the second curve and the fat line. 
        FatLine fatLine;
        NormalizedImplicitLine implicitLine;
        bool foundIntersections;
        
        if (clip_curve.orderOne) {
            implicitLine =  normalized_implicit(clip_curve.curve_data.curve_order_one);
            fatLine = fat_line(implicitLine, clip_curve.curve_data.curve_order_one);
            foundIntersections = intersection_points(other_curve.curve_data.curve_order_two, fatLine, intersections); 
        } else {
            implicitLine =  normalized_implicit(clip_curve.curve_data.curve_order_two);
            fatLine = fat_line(implicitLine, clip_curve.curve_data.curve_order_two);
            foundIntersections = intersection_points(other_curve.curve_data.curve_order_one, fatLine, intersections); 
        }

        Float* pointsPtr = intersections.get_host_ptr(); 
        if (!foundIntersections) {
            if (param_stack.size() == 0) 
                break; 

            clip_curve = param_stack.top();
            param_stack.pop(); 
            other_curve = param_stack.top(); 
            param_stack.pop(); 
            continue;
        }
        
        t_min = pointsPtr[0];
        t_max = pointsPtr[1];
        Float t_range = t_max - t_min; 
    
        // If we managed to clip less than 20% of the interval, there are probably two intersections. Split the curve into two 
        // and recursively call intersect() on the two halves.
        if (t_range > 0.80) {
            CurveStruct segment_one;
            segment_one.orderOne = other_curve.orderOne; 
            int i = 0; 
            
            // Copy the curve to segment_one.
            if (other_curve.orderOne) {
                Curve<p_order1> segment_onePoints;
                for (auto &pt : segment_onePoints.components())
                    pt = other_curve.curve_data.curve_order_one[i++];
                CurveData segment_oneData; 
                segment_oneData.curve_order_one = segment_onePoints;
                segment_one.curve_data = segment_oneData;

                // Make segment one the first half of the curve.
                dray::DeCasteljau::split_inplace_left<Curve<p_order1>>(
                    segment_one.curve_data.curve_order_one, 
                    (Float) t_max, 
                    other_curve.size() - 1
                );
                // Make other_curve the second half of the curve.
                dray::DeCasteljau::split_inplace_right<Curve<p_order1>>(
                    other_curve.curve_data.curve_order_one, 
                    (Float) t_min / t_max, 
                    other_curve.size() - 1
                );
            } else {
                Curve<p_order2> segment_onePoints;
                for (auto &pt : segment_onePoints.components())
                    pt = other_curve.curve_data.curve_order_two[i++];
                CurveData segment_one_data; 
                segment_one_data.curve_order_two = segment_onePoints;
                segment_one.curve_data = segment_one_data;

                // Make segment one the first half of the curve.
                dray::DeCasteljau::split_inplace_left<Curve<p_order2>>(
                    segment_one.curve_data.curve_order_two, 
                    (Float) t_max, 
                    other_curve.size() - 1 
                );
                // Make other_curve the second half of the curve.
                dray::DeCasteljau::split_inplace_right<Curve<p_order2>>(
                    other_curve.curve_data.curve_order_two, 
                    (Float) t_min / t_max, 
                    other_curve.size() - 1 
                );
            }

            param_stack.push(other_curve); 
            param_stack.push(clip_curve);  
            other_curve = segment_one;
            continue; 
        }  

        // Update the t_max and clip at t_max.
        Float interval_length = (other_curve.t_max - other_curve.t_min);
        other_curve.t_max -= (1 - t_max) * (interval_length);
        // Update the t_min and clip at t_min (we have to adjust t_min because we shortened the interval).
        t_min = t_min / t_max;
        interval_length = (other_curve.t_max - other_curve.t_min);
        other_curve.t_min += t_min * interval_length; 

        // Base case.
        if ((other_curve.t_max - other_curve.t_min) < threshold) {
            res.resize(res.size() + 1);

            Float *resPtr = res.get_host_ptr();    
            resPtr[res.size() - 1] = (other_curve.t_max + other_curve.t_min) / 2; 

            if (param_stack.size() == 0) 
                ++num_intersections;
                break; 
            
            clip_curve  = param_stack.top(); 
            param_stack.pop();
            other_curve = param_stack.top(); 
            param_stack.pop(); 
            continue; 
        }
        
        if (other_curve.orderOne) { 
            dray::DeCasteljau::split_inplace_left<Curve<p_order1>>(
                other_curve.curve_data.curve_order_one, 
                (Float) t_max, 
                other_curve.size() - 1);
            dray::DeCasteljau::split_inplace_right<Curve<p_order1>>(
                other_curve.curve_data.curve_order_one, 
                (Float) t_min, 
                other_curve.size() - 1);
        } else {
            dray::DeCasteljau::split_inplace_left<Curve<p_order2>>(
                other_curve.curve_data.curve_order_two, 
                (Float) t_max, 
                other_curve.size() - 1);
            dray::DeCasteljau::split_inplace_right<Curve<p_order2>>(
                other_curve.curve_data.curve_order_two, 
                (Float) t_min,
                other_curve.size() - 1);
        }
    
        CurveStruct temp = other_curve; 
        other_curve = clip_curve;
        clip_curve  = temp; 
    }

    return num_intersections > 0; 
}

template <uint32 p_order>
FatLine fat_line(NormalizedImplicitLine l, Curve<p_order> curve_one) {
    size_t degree   = curve_one.size() - 1;  
    
    Float minDist = l.dist(curve_one[0]);  
    Float maxDist = l.dist(curve_one[0]);
   
    // We can use tight bounds in the case where the curve is quadratic.
    if (degree == 2) {
        Vec2D p = curve_one[1]; 
        Float scaled_p_dist = l.dist(p) / 2;
        minDist = std::min(Float(0), scaled_p_dist);
        maxDist = std::max(Float(0), scaled_p_dist);  
    } else if (degree == 3) { 
        // We can user tigher bounds in the case where the curve is cubic.
        Vec2D p1 = curve_one[1];
        Vec2D p2 = curve_one[2];
        Float p1_dist = l.dist(p1); 
        Float p2_dist = l.dist(p2); 
        if (p1_dist * p2_dist > 0) {
            minDist = (3./4) * std::min(std::min(p1_dist, p2_dist), Float(0));
            maxDist = (3./4) * std::max(std::max(p1_dist, p2_dist), Float(0));
        } else {
            minDist = (4. / 9) * std::min(std::min(p1_dist, p2_dist), Float(0));
            maxDist = (4. / 9) * std::max(std::max(p1_dist, p2_dist), Float(0));
        }
    } else {
        for (size_t i = 1; i < degree + 1; ++i) {
            Vec2D p = curve_one[i];
            Float dist = l.dist(p); 
            if (dist < minDist)
                minDist = dist;
            if (dist > maxDist)
                maxDist = dist;
        }
    }

    FatLine fatLine        = FatLine{};
    fatLine.upper_bound     = maxDist;
    fatLine.lower_bound     = minDist;
    fatLine.line           = l; 
    return fatLine;
}

bool t_intersection(FatLine& line,
                    Vec2D& prev_point,
                    Vec2D& curr_point, 
                    Float& t_min, 
                    Float& t_max, 
                    Float& t_0, 
                    Float& t_1, 
                    Float& t_interval) {

    Float y_0  = line.line.dist(prev_point); 
    Float y_1  = line.line.dist(curr_point);
    t_1 += t_interval;

    // Create a line segment by looking from the (i)th point to the (i-1)th point.
    // This starts at 'prev_point' and goes to the currPoint. 
    Float slope = (y_1 - y_0) / (t_1 - t_0); 
    Float delta_t; 

    bool foundIntersection = false; 
    bool cond_one = (y_0 < line.upper_bound && line.upper_bound < y_1);
    bool cond_two = (y_1 < line.upper_bound && line.upper_bound < y_0);
    
    if (cond_one || cond_two) { 
        delta_t = (line.upper_bound - y_0) / slope; 
        Float intersection_t = t_0 + delta_t; 
        t_min   = std::min(t_min, intersection_t);
        t_max   = std::max(t_max, intersection_t);
        foundIntersection = true; 
    } 

    cond_one = (y_0 < line.lower_bound && line.lower_bound < y_1);
    cond_two = (y_1 < line.lower_bound && line.lower_bound < y_0);

    if (cond_one || cond_two) { 
        delta_t = (line.lower_bound - y_0) / slope;  
        Float intersection_t = t_0 + delta_t; 
        t_min   = std::min(t_min, intersection_t);
        t_max   = std::max(t_max, intersection_t);
        foundIntersection = true; 
    } 

    // Update for next iteration.
    prev_point = curr_point; 
    t_0       = t_1;

    return foundIntersection; 
}

// Takes a bezier curve and creates a normalized implicit line through it.
template <uint32 p_order>
NormalizedImplicitLine normalized_implicit(Curve<p_order> curve) {
    // First, create a line by looking at the first and last control point
    Vec2D first_control_point = curve[0]; 
    size_t size = curve.size();
    Vec2D last_control_point = curve[size - 1];
    Vec2D difference = last_control_point - first_control_point;
    // Get the orthogonal vector in 2D 
    return to_normalized_implicit(difference, first_control_point); 
}

// Takes a vector and creates a normalized implicit line from it. 
NormalizedImplicitLine to_normalized_implicit(Vec2D original_line, Vec2D p_0) {
    // Get the orthogonal vector in 2D 
    Vec2D normal_vect = {-1 * original_line[1], original_line[0]}; 
    normal_vect.normalize();

    NormalizedImplicitLine line = NormalizedImplicitLine{};
    line.N = normal_vect;
    line.P_0 = p_0;
    return line; 
}

// ================================
// === Ray-Surface Intersection ===
// ================================

/* 
    We represent the ray as an intersection of two planes.
*/
void rayToIntersectingPlanes(Vec3D &ray_direction, Vec3D &plane_one_normalVec, Vec3D &plane_two_normalVec) {
    plane_one_normalVec = {-ray_direction[1], ray_direction[0], 0};
    plane_two_normalVec = {0, -ray_direction[2], ray_direction[1]};
}

/* 
    We project the mesh two dimensions. This new surface is also a Bezier surface, so we just 
    compute the new set of control points.

    Assume that we have control point p_ij at row i and column j. Further,
    0 <= i < n, 
    0 <= j < m,
    so that we have 'm' columns and 'n' rows. Then the index is
    Then p_ij = controlPoints[i * m + j]

    Note that we are not using rational Bezier curves, so the weights are all 1. 
*/ 
void projectTo2D(
    Vec3D &plane_one_normal,
    Vec3D &plane_two_normal, 
    Vec3D &rayOrigin, 
    Array<Vec3D> &controlPoints,
    Array<Vec2D> &new_control_pts,
    size_t n,
    size_t m) {
    /* 
    Adjust the size of the results array to have 
    as many points as the current number of control 
    points.
    */ 
    new_control_pts.resize(controlPoints.size());
    Vec2D *resPtr = new_control_pts.get_host_ptr(); 
    Vec3D *controlPointsPtr = controlPoints.get_host_ptr(); 

    Float h_ij1 = 0;
    Float h_ij2 = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            size_t index = (i * m) + j;
            h_ij1 = dray::dot(plane_one_normal, controlPointsPtr[index]) - 
                    dray::dot(plane_one_normal, rayOrigin);
            h_ij2 = dray::dot(plane_two_normal, controlPointsPtr[index]) - 
                    dray::dot(plane_two_normal, rayOrigin);
            resPtr[index] = {h_ij1, h_ij2}; 
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
        Array<Vec2D> &new_control_pts,
        FatLine &Lu,
        FatLine &Lv,
        size_t n,
        size_t m) 
{
    Vec2D* controlPointsPtr = new_control_pts.get_host_ptr();
    const Vec2D r_00 = controlPointsPtr[0];
    const Vec2D r_n0 = controlPointsPtr[n * m];
    const Vec2D r_0m = controlPointsPtr[m - 1];
    const Vec2D r_nm = controlPointsPtr[(n * m) + (m - 1)];
    
    Vec2D origin = {0, 0}; 
    Vec2D l1 = (r_n0 - r_00) + (r_nm - r_0m);
    Vec2D l2 = (r_0m - r_00) + (r_nm - r_n0); 

    NormalizedImplicitLine l1_implicit = to_normalized_implicit(l1, origin);
    NormalizedImplicitLine l2_implicit = to_normalized_implicit(l2, origin); 

    Lu.line = l1_implicit;
    Lv.line = l2_implicit; 
    Lu.upper_bound = 0;
    Lu.lower_bound = 0; 
    Lv.upper_bound = 0;
    Lv.lower_bound = 0;
}

// Assume that we have control point p_ij at row i and column j. Further,
// 0 <= i < n, 
// 0 <= j < m,
// so that we have 'm' columns and 'n' rows. Then the index is
// Then p_ij = controlPoints[i * m + j]
// TODO: check
void getDistanceControlPoints(Array<Vec2D> &distanceControlPoints, FatLine fatLine, 
                              Array<Vec2D> controlPoints, int n, int m, bool directionU) { 
    distanceControlPoints.resize(n*m);
    Vec2D* distanceCtrlPtr = distanceControlPoints.get_host_ptr();
    Vec2D* controlPtr      = controlPoints.get_host_ptr(); 

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; j++) { 
            int index = (i * m) + j; 
            // Compute the distance to the fat line. This is the 'y' coordinate. 
            Float dist = fatLine.line.dist(controlPtr[index]);

            // Compute the x-coord. Note that the control points D_ij are evenly spaced 
            // in the direction of the line. Thus if we are splitting along u, 
            // u_ij = i/n.  
            // TODO: check
            Float x_coord = 0; 
            if (directionU)
                x_coord = i / n; 
            else 
                x_coord = j / m; 

            // Store the new control point.
            distanceCtrlPtr[index] = {x_coord, dist};
        }
    }
}

void getConvexHull(Array<Vec2D> convexHull, Array<Vec2D> controlPoints, int n, int m, bool directionU) { 
    if (directionU) 
        convexHull.resize(0); // TODO 
    else
        convexHull.resize(0); // TODO

    Vec2D* controlPtr = controlPoints.get_host_ptr(); 
    for (int i = 0; i < n; ++i) { 
        for (int j = 0; j < m; ++j) { 
            int index = (i * m) + j; 
            Vec2D currPoint = controlPtr[index]; 
            Float dist = currPoint[1]; 
        }
    }
}

bool intersectMesh(size_t n, size_t m,
                   Ray ray,  
                   Array<Vec3D> controlPoints, 
                   size_t maxIterations = 10) {
    Vec3D plane_one_normal;
    Vec3D plane_two_normal;
    rayToIntersectingPlanes(
        ray.m_dir,
        plane_one_normal, 
        plane_two_normal);

    Array<Vec2D> new_control_pts; 
    projectTo2D(plane_one_normal, plane_two_normal, ray.m_orig,
                controlPoints, new_control_pts, n, m);
   
    FatLine Lu;
    FatLine Lv;
    getFatLines(new_control_pts, Lu, Lv, n, m); 
    bool splittingU = true;
    while (true) {
        if (--maxIterations < 0) 
            break;

        // Always clip Lu (we swap Lu and Lv at the end of the loop).
        Array<Vec2D> distance_control_pts;
        getDistanceControlPoints(distance_control_pts, Lu, new_control_pts, n, m, splittingU); 
        
        // Now get the convex hull.
        Array<Vec2D> convex_hull; 
        getConvexHull(convex_hull, distance_control_pts, n, m, splittingU);

        // Swap the two lines (clip in the other direction) 
        FatLine temp = Lu; 
        Lu = Lv; 
        Lv = temp; 
        splittingU = !splittingU;
    }
}

// ==============================
// === Explicit instantiation ===
// ==============================
template bool intersect(
            Array<Float> &res, 
            Curve<3> &curve_one,
            Curve<3> &curve_two,
            int maxIterations,
            Float threshold, 
            bool returnClosestSolution);
template bool intersect(
            Array<Float> &res, 
            Curve<1> &curve_one,
            Curve<1> &curve_two,
            int maxIterations,
            Float threshold,
            bool returnClosestSolution);

// Try templating on two different orders. 
// TODO: remove
template bool intersect(
            Array<Float> &res, 
            Curve<2> &curve_one,
            Curve<1> &curve_two,
            int maxIterations,
            Float threshold,
            bool returnClosestSolution);

template NormalizedImplicitLine normalized_implicit(Curve<2> curve);

template FatLine fat_line(NormalizedImplicitLine l, Curve<1> curve_one);
template FatLine fat_line(NormalizedImplicitLine l, Curve<2> curve_one);
template FatLine fat_line(NormalizedImplicitLine l, Curve<4> curve_one);

template bool intersection_points(Curve<2> curve, 
                                  FatLine line, 
                                  Array<Float> &intersections);
    

}
}
