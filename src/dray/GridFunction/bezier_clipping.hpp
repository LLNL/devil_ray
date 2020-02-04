#ifndef DRAY_BEZIER_CLIPPING_HPP
#define DRAY_BEZIER_CLIPPING_HPP

#include <dray/types.hpp>
/*
    This implementation is based on the paper   
    "Curve Intersection using Bezier Clipping"
    by T W Sederberg and T Nishita 
*/ 

namespace dray 
{

namespace bezier_clipping 
{

// A NormalizedImplicitLine is one of the form 
// <N, (x - p_0)> = 0
// i.e. the dot product of N and (x - p_o), where 
// p_o is one of the points on the line. Further,
// the norm of N equals 1: ||N|| = 1. 
template <class Point>
struct NormalizedImplicitLine {
    // Computes the distance to an arbitrary point. 
    Float dist(Point point);

    Point N;
    Point P_0;

    friend std::ostream& operator<<(std::ostream& out, const NormalizedImplicitLine &line) {
        return out << "N: " << line.N << " P_0: " << line.P_0;
    } 
};

// A FatLine is a NoramlizedImplicitLine with bounds around it. 
template <class Point>
struct FatLine {
    Float upperBound;
    Float lowerBound;  

    NormalizedImplicitLine<Point> line;

    friend std::ostream& operator<<(std::ostream& out, const FatLine<Point> &line) {
        return out << "Upper Bound: " << line.upperBound << " Lower Bound: " << line.lowerBound;
    } 
};

template <class Point>
Float NormalizedImplicitLine<Point>::dist(Point point) { 
    // If the line is of the form 
    // N*X - N*P_0 = 0 
    // Then the distance from an arbitrary point (x, y) 
    // is d(x, y) = ax + by + c
    Float signedDistance = dot(N, point) - dot(N, P_0);
    return signedDistance;   
}

// Take two bezier curves, find their intersection
template <class Curve, class Point>
Point intersect(Curve curveOne, Curve curveTwo, int maxIterations = 10, float threshold = 1e-3) {
    if (maxIterations == 0) {
        return nullptr;
    }
    --maxIterations;

    NormalizedImplicitLine<Point> line    = normalized_implicit(curveOne);
    FatLine<Point>                fatLine = fat_line(line, curveOne);

    // Find the points of intersection with between the second curve and 
    // the fat line. 
    Array<Point> intersections;
    bool foundIntersections = intersection_points(curveTwo, fatLine, intersections); 
    Point* pointsPtr = intersections.get_host_ptr(); 
    if (!foundIntersections) {
        return nullptr;
    }
    
    // Split the second curve at those points of intersection
    Curve segmentOne, segmentTwo, segmentThree; 
    segmentOne, segmentTwo   = de_casteljau(curveTwo,   pointsPtr[0]);
    segmentTwo, segmentThree = de_casteljau(segmentTwo, pointsPtr[1]);

    // Repeat by clipping THE OTHER curve. 
    return intersect(segmentTwo, curveOne);
}

template <class Point> 
bool t_intersection(FatLine<Point>& line, Point& prevPoint, Point& currPoint, Float& t_min, Float& t_max, Float& t_0, Float& t_1, Float& t_interval) { 
    Float y_1  = line.line.dist(currPoint); 
    Float y_0  = line.line.dist(prevPoint); 
    t_1 += t_interval;  
    
    // Create a line segment by looking from the (i)th point to the (i-1)th point.
    // This starts at 'prevPoint' and goes to the currPoint. 
    Float slope = (y_1 - y_0) / (t_1 - t_0); 
    
    Float delta_t; 
    bool foundIntersection = false; 

    if (y_0 < line.upperBound < y_1) { 
        // y = y_0 + slope * (delta_t) 
        // (y - y_0) / slope = delta_t
        delta_t = (line.upperBound - y_0) / slope; 
        t_min   = std::min(t_min, t_0 + delta_t);
        t_max   = std::max(t_max, t_0 + delta_t);
        foundIntersection = true; 
    }

    if (y_0 < line.lowerBound < y_1) { 
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

// TODO: template on dimension instead of Point?
template <class Curve, class Point> 
bool intersection_points(Curve curve, FatLine<Point> line, Array<Point> &intersections) {
    // At most two intersections are possible. 
    intersections.resize(2); 
    Point* pointsPtr = intersections.get_host_ptr();    

    bool foundIntersection = false; 

    // Assume that the control points are in order.
    Point* curvePtr   = curve.get_host_ptr();
    size_t numControlPoints = curve.size(); 
    Point prevPoint         = curvePtr[0];
    Point currPoint; 
    Float t_interval        = (1 / (numControlPoints - 1));
    Float t_0 = 0; 
    Float t_1 = 0; 
    Float t_min = 0; 
    Float t_max = 0; 
    // Iterate through the control points. 
    for (size_t i = 1; i < numControlPoints; ++i) {
        currPoint = curvePtr[i];
        foundIntersection = t_intersection(line, prevPoint, currPoint, t_min, t_max, t_0, t_1, t_interval); 
    }

    // Consider the line segment from P_0 to last control point. 
    t_0 = 0; 
    t_1 = 0; 
    foundIntersection = t_intersection(line, curvePtr[0], curvePtr[numControlPoints - 1], t_min, t_max, t_0, t_1, t_interval);
    
    if (foundIntersection) {   
        
        pointsPtr[0] = curvePtr[size_t(std::round(t_min / t_interval))]; 
        pointsPtr[1] = curvePtr[size_t(std::round(t_max / t_interval))]; 
    }

    return foundIntersection; 
}

// Takes a bezier curve and creates a normalized implicit line through it.
template <class Curve, class Point>
NormalizedImplicitLine<Point> normalized_implicit(Curve curve) {
    // First, create a line by looking at the first and last control point
    const Point* curvePtr = curve.get_host_ptr_const();  
    Point firstControlPoint = curvePtr[0]; 
    size_t size = curve.size(); 
    Point lastControlPoint = curvePtr[size - 1];

    Point difference = lastControlPoint - firstControlPoint;; 

    // Sample vectors from the space until they are not colinear (TEMP). 
    Point someVect;
    do {
        int dims = someVect.size(); 
        for (int i = 0; i < dims; ++i) {
            someVect[i] = (rand() % 10) + 1; 
        }
    } while (abs( dot(someVect, difference) ) < 1e-3);

    Point normalVect = cross(someVect, difference);

    NormalizedImplicitLine<Point> line{};
    line.N = normalVect;
    line.P_0 = firstControlPoint;
    return line; 
}

// // fat_line creates a FatLine from a bezier curve and a NormalizedImplicitLine.
// // 
// // This is done by taking the maximum (signed) distance from each control point to 
// // the NormalizedImplicitLine.
template <class Curve, class Point>
FatLine<Point> fat_line(NormalizedImplicitLine<Point> l, Curve curveOne) {
    const Point* curveOnePtr = curveOne.get_host_ptr_const();  
    size_t size   = curveOne.size();  

    // the degree of a Bezier curve is one less than the number of control points 
    size_t degree = size - 1; 
    
    Float minDist = l.dist(curveOnePtr[0]);  
    Float maxDist = l.dist(curveOnePtr[0]);
   
    // We can use tight bounds in the case where the curve is quadratic
    if (degree == 2) { 
        Point p = curveOnePtr[1]; 
        Float scaled_p_dist = l.dist(p) / 2;
        minDist = std::min(Float(0), scaled_p_dist);
        maxDist = std::max(Float(0), scaled_p_dist);  
    } else if (degree == 3) { 
        // We can user tigher bounds in the case where the curve is cubic 
        Point p1 = curveOnePtr[1];
        Point p2 = curveOnePtr[2];
        Float p1_dist = l.dist(p1);
        Float p2_dist = l.dist(p2);
        minDist = (4 / 9) * std::min(std::min(p1_dist, p2_dist), Float(0));
        maxDist = (4 / 9) * std::max(std::max(p1_dist, p2_dist), Float(0));
    } else {
        for (size_t i = 1; i < size; ++i) {
            Point p = curveOnePtr[i];
            Float dist = l.dist(p); 
            if (dist < minDist) {
                minDist = dist;
            }
            if (dist > maxDist) {
                maxDist = dist;
            }
        }
    }

    FatLine<Point> fatLine = FatLine<Point>{};
    fatLine.upperBound     = maxDist;
    fatLine.lowerBound     = minDist;
    fatLine.line           = l; 
    return fatLine;
}

}
}
#endif // DRAY_BEZIER_CLIPPING_HPP
