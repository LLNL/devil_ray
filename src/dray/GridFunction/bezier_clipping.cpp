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
template <uint32 p_order> 
bool intersection_points(Curve<p_order> curve, FatLine line, Range& t_range)
{
    bool found_intersection  = false; 

    // Assume that the control points are in order.
    Float prev_y = line.dist(curve[0]);  
    Float curr_y;

    // Compute the distance between each control point in the 't' axis. 
    Float t_interval = (1. / (curve.size()  - 1));
    // Minimum and maximum t values for intersection range.
    t_range.reset(); 

    // Iterate through the control points. 
    for (int i = 1; i < curve.size(); ++i) {
        // Find the intersection between the line from current control point to previous control point
        // and the lines bounding the fat line.
        curr_y  = line.dist(curve[i]);
        found_intersection = t_intersection(line, prev_y, curr_y, t_range, (i - 1) * t_interval,
                                            i * t_interval) || found_intersection; 
        prev_y = curr_y; 
    }

    // Consider the line segment from P_0 to last control point.
    found_intersection = t_intersection(line, line.dist(curve[0]), curr_y, t_range, 0., 1.)
                         || found_intersection;

    return found_intersection; 
}

template <uint32 p_order1, uint32 p_order2>
union CurveData {
    Curve<p_order1> curve_order_one; 
    Curve<p_order2> curve_order_two;
};

template <uint32 p_order1, uint32 p_order2> 
struct CurveStruct { 
    bool order_one;
    CurveData<p_order1, p_order2> curve_data; 
    Range t_range;

    int size() { 
        if (order_one)
            return curve_data.curve_order_one.size();

        return curve_data.curve_order_two.size();
    }
};

template <uint32 p_order1, uint32 p_order2>
int intersect(Array<Float> &res, Curve<p_order1> &curve_one, Curve<p_order2> &curve_two,
               int max_iterations = 10, Float threshold = 0.01f)
{    
    using CurveData   = CurveData<p_order1, p_order2>; 
    using CurveStruct = CurveStruct<p_order1, p_order2>; 
    stack<CurveStruct> param_stack;
    Range t_range; 

    // Initialize the CurveStructs.
    CurveStruct clip_curve   = {true,  CurveData{.curve_order_one = curve_one}, Range::ref_universe()}; 
    CurveStruct other_curve  = {false, CurveData{.curve_order_two = curve_two}, Range::ref_universe()};
    size_t num_intersections = 0; 

    while (true) {
        if (--max_iterations < 0)
            return 0;
        
        // Find the points of intersection with between the second curve and the fat line. 
        FatLine implicit_line;
        bool found_intersections;
        
        if (clip_curve.order_one) {
            implicit_line =  normalized_implicit(clip_curve.curve_data.curve_order_one);
            fat_line_bounds(implicit_line, clip_curve.curve_data.curve_order_one);
            found_intersections = intersection_points(other_curve.curve_data.curve_order_two, 
                                                      implicit_line, t_range); 
        } else {
            implicit_line =  normalized_implicit(clip_curve.curve_data.curve_order_two);
            fat_line_bounds(implicit_line, clip_curve.curve_data.curve_order_two);
            found_intersections = intersection_points(other_curve.curve_data.curve_order_one,
                                                      implicit_line, t_range); 
        }

        if (!found_intersections) {
            if (param_stack.size() == 0) 
                break; 

            clip_curve = param_stack.top();
            param_stack.pop(); 
            other_curve = param_stack.top(); 
            param_stack.pop(); 
            continue;
        }
    
        // If we managed to clip less than 20% of the interval, there are probably two intersections.
        // Split the curve into two and recursively call intersect() on the two halves.
        if (t_range.length() > 0.80) {
            CurveStruct segment_one = {other_curve.order_one};
            
            // Copy the curve to segment_one.
            if (other_curve.order_one) {
                Curve<p_order1> segment_one_points = create_curve(other_curve.curve_data.curve_order_one);
                segment_one.curve_data = CurveData{segment_one_points};

                // Make segment one the first half of the curve.
                dray::DeCasteljau::split_inplace_left<Curve<p_order1>>(
                    segment_one.curve_data.curve_order_one, 
                    t_range.max(), other_curve.size() - 1);
                // Make other_curve the second half of the curve.
                dray::DeCasteljau::split_inplace_right<Curve<p_order1>>(
                    other_curve.curve_data.curve_order_one, 
                    t_range.min() / t_range.max(), other_curve.size() - 1);
            } else {
                Curve<p_order2> segment_one_points = create_curve(other_curve.curve_data.curve_order_two);
                segment_one.curve_data = CurveData{.curve_order_two = segment_one_points};

                dray::DeCasteljau::split_inplace_left<Curve<p_order2>>(
                    segment_one.curve_data.curve_order_two, 
                    t_range.max(), other_curve.size() - 1);
                dray::DeCasteljau::split_inplace_right<Curve<p_order2>>(
                    other_curve.curve_data.curve_order_two, 
                    t_range.min() / t_range.max(), other_curve.size() - 1);
            }

            param_stack.push(other_curve); 
            param_stack.push(clip_curve);  
            other_curve = segment_one;
            continue; 
        }  

        // Update the t_max and clip at t_max.
        other_curve.t_range.update(other_curve.t_range.min(),
                                   other_curve.t_range.max() - 
                                   (1 - t_range.max()) * other_curve.t_range.length());
        // Update the t_min and clip at t_min (we have to adjust t_min because we shortened the interval).
        t_range.update(t_range.min() / t_range.max(), t_range.max());
        other_curve.t_range.update(other_curve.t_range.min() + 
                                   t_range.min() * other_curve.t_range.length(),
                                   other_curve.t_range.max());

        // Base case.
        if (other_curve.t_range.length() < threshold) {
            ++num_intersections;
            res.resize(res.size() + 1);

            Float *resPtr = res.get_host_ptr();    
            resPtr[res.size() - 1] = other_curve.t_range.center();

            if (param_stack.size() == 0) 
                break; 
            
            clip_curve  = param_stack.top(); 
            param_stack.pop();
            other_curve = param_stack.top(); 
            param_stack.pop(); 
            continue; 
        }
        
        if (other_curve.order_one) { 
            dray::DeCasteljau::split_inplace_left<Curve<p_order1>>(
                other_curve.curve_data.curve_order_one, 
                t_range.max(), other_curve.size() - 1);
            dray::DeCasteljau::split_inplace_right<Curve<p_order1>>(
                other_curve.curve_data.curve_order_one, 
                t_range.min(), other_curve.size() - 1);
        } else {
            dray::DeCasteljau::split_inplace_left<Curve<p_order2>>(
                other_curve.curve_data.curve_order_two, 
                t_range.max(), other_curve.size() - 1);
            dray::DeCasteljau::split_inplace_right<Curve<p_order2>>(
                other_curve.curve_data.curve_order_two, 
                t_range.min(), other_curve.size() - 1);
        }
    
        CurveStruct temp = other_curve; 
        other_curve = clip_curve;
        clip_curve  = temp; 
    }

    return num_intersections; 
}

template <uint32 p_order>
void fat_line_bounds(FatLine &l, Curve<p_order> curve_one)
{
    size_t degree   = curve_one.size() - 1;  
    
    Float min_dist = l.dist(curve_one[0]);  
    Float max_dist = l.dist(curve_one[0]);
   
    // We can use tight bounds in the case where the curve is quadratic or cubic.
    if (degree == 2) {
        Vec2D p = curve_one[1]; 
        Float scaled_p_dist = l.dist(p) / 2;
        min_dist = std::min(Float(0), scaled_p_dist);
        max_dist = std::max(Float(0), scaled_p_dist);  
    } else if (degree == 3) { 
        Float p1_dist = l.dist(curve_one[1]); 
        Float p2_dist = l.dist(curve_one[2]); 
        if (p1_dist * p2_dist > 0) {
            min_dist = (3./4) * std::min(std::min(p1_dist, p2_dist), Float(0));
            max_dist = (3./4) * std::max(std::max(p1_dist, p2_dist), Float(0));
        } else {
            min_dist = (4. / 9) * std::min(std::min(p1_dist, p2_dist), Float(0));
            max_dist = (4. / 9) * std::max(std::max(p1_dist, p2_dist), Float(0));
        }
    } else {
        for (size_t i = 1; i < degree + 1; ++i) {
            Float dist = l.dist(curve_one[i]);
            min_dist = std::min(dist, min_dist); 
            max_dist = std::max(dist, max_dist);
        }
    }

    l.bound.update(min_dist, max_dist);
}

bool t_intersection(FatLine& line, Float y_0, Float y_1, Range& t_range, Float t_0, Float t_1)
{
    // Create a line segment by looking at the two points.
    Float slope = (y_1 - y_0) / (t_1 - t_0); 
    
    Float delta_t; 

    bool found_intersection = false; 

    if ((y_0 < line.bound.max() && line.bound.max() < y_1) || 
        (y_1 < line.bound.max() && line.bound.max() < y_0)) { 
        delta_t = (line.bound.max() - y_0) / slope; 
        Float intersection_t = t_0 + delta_t; 
        t_range.update(std::min(t_range.min(), intersection_t),
                       std::max(t_range.max(), intersection_t));
        found_intersection = true; 
    } 

    if ((y_0 < line.bound.min() && line.bound.min() < y_1) ||
        (y_1 < line.bound.min() && line.bound.min() < y_0)) { 
        delta_t = (line.bound.min() - y_0) / slope;  
        Float intersection_t = t_0 + delta_t; 
        t_range.update(std::min(t_range.min(), intersection_t),
                       std::max(t_range.max(), intersection_t));
        found_intersection = true; 
    } 

    // Update for next iteration.
    return found_intersection; 
}

// Takes a bezier curve and creates a normalized implicit line through it.
template <uint32 p_order>
FatLine normalized_implicit(Curve<p_order> curve)
{  
    Vec2D original_line = curve[curve.size() - 1] - curve[0];
    Vec2D normal_vect = {-1 * original_line[1], original_line[0]}; 
    normal_vect.normalize();
    return FatLine{normal_vect, curve[0]};
}

// ==============================
// === Explicit instantiation ===
// ==============================
template int intersect(Array<Float> &res, Curve<1> &curve_one,
                        Curve<1> &curve_two, int max_iterations, Float threshold);
template int intersect(Array<Float> &res, Curve<2> &curve_one,
                        Curve<1> &curve_two, int max_iterations, Float threshold);
template int intersect(Array<Float> &res, Curve<3> &curve_one,
                        Curve<3> &curve_two, int max_iterations, Float threshold);

template FatLine normalized_implicit(Curve<2> curve);
template void fat_line_bounds(FatLine& l, Curve<4> curve_one);
template bool intersection_points(Curve<2> curve, FatLine line, Range& t_range);
}
}