#ifndef DRAY_BEZIER_CLIPPING_HPP
#define DRAY_BEZIER_CLIPPING_HPP

#include <dray/types.hpp>

namespace dray 
{

namespace bezier_clipping 
{

struct FatLine {
    Float upperBound;
    Float lowerBound;  

    friend std::ostream& operator<<(std::ostream& out, const FatLine &line) {
        return out << "Upper Bound: " << line.upperBound << " Lower Bound: " << line.lowerBound;
    } 
};


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

    NormalizedImplicitLine<Point> line = normalized_implicit(curveOne);
    FatLine                       f    = fat_line(line, curveOne);

    // Find the points of intersection with between the second curve and 
    // the fat line. 

    Point* points = intersection_points(curveTwo, f); 
    if (len(points) == 0) {
        return nullptr;
    }
    
    // Split the second curve at those points of intersection
    Curve segmentOne, segmentTwo, segmentThree; 
    segmentOne, segmentTwo   = de_casteljau(curveTwo,   points[0]);
    segmentTwo, segmentThree = de_casteljau(segmentTwo, points[1]);

    // Repeat by clipping THE OTHER curve. 
    return intersect(segmentTwo, curveOne);
}

template <class Curve, class Point>
Point* intersection_points(Curve curve, FatLine line) {
    // TODO
    // Point[] points;
    // return points;
    return nullptr; 
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

    cout << difference << endl; 
 
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
FatLine fat_line(NormalizedImplicitLine<Point> l, Curve curveOne) {
    const Point* curveOnePtr = curveOne.get_host_ptr_const();  
    size_t size = curveOne.size(); 
    
    Float minDist = l.dist(curveOnePtr[0]);  
    Float maxDist = l.dist(curveOnePtr[0]);
   
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
    FatLine fatLine = FatLine{};
    fatLine.upperBound = maxDist;
    fatLine.lowerBound = minDist;
    return fatLine;
}

}
}
#endif // DRAY_BEZIER_CLIPPING_HPP
