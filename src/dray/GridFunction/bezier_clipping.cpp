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

template <class Curve> 
bool intersection_points(Curve curve, FatLine line, Array<Float> &intersections) {
    cout << "Starting Intersection_Points" << endl; 
    // At most two intersections are possible. 
    intersections.resize(2); 
    Float* intersectionsPtr = intersections.get_host_ptr();    
    bool foundIntersection = false; 

    // Assume that the control points are in order.
    Point* curvePtr         = curve.get_host_ptr();
    size_t numControlPoints = curve.size(); 

    cout << "numControlPoints: " << numControlPoints << endl; 

    Point prevPoint         = curvePtr[0];
    Point currPoint; 

    // Compute the distance between each control point in the 't' axis. 
    Float t_interval        = (1. / (numControlPoints - 1));
    Float t_0 = 0;   // t value for previous control point
    Float t_1 = 0;   // t value for current control point
    Float t_min = std::numeric_limits<Float>::infinity(); // minimum t value for intersection range 
    Float t_max = -1 * std::numeric_limits<Float>::infinity(); // maximum t value for intersection range

    cout << "Beginning to iterate through the control points" << endl; 

    // Iterate through the control points. 
    for (size_t i = 1; i < numControlPoints; ++i) {
        currPoint = curvePtr[i];
        // Find the intersection between the line from current control point to previous control point
        // and the lines bounding the fat line.
        bool intersect = t_intersection(line, prevPoint, currPoint, t_min, t_max, t_0, t_1, t_interval);
        foundIntersection = foundIntersection || intersect; 
    }

    // Consider the line segment from P_0 to last control point. 
    t_0 = 0; 
    t_1 = 0; 
    t_interval = Float(1); 
    bool intersect = t_intersection(line, curvePtr[0], curvePtr[numControlPoints - 1], t_min, t_max, t_0, t_1, t_interval);
    foundIntersection = foundIntersection || intersect; 
    
    if (foundIntersection) {           
        intersectionsPtr[0] = t_min; 
        intersectionsPtr[1] = t_max; 
    }

    cout << ">>> Found Intersection <<< " << foundIntersection << endl; 
    cout << "t_min: " << t_min << " t_max: " << t_max << endl; 

    return foundIntersection; 
}

template <class Curve> 
void de_casteljau(Curve &resOne, Curve &resTwo, Curve input, Float t) {
    cout << "FIX ME!" << endl; 
    resTwo = input; 
    resOne = input; 

    cout << "Creating the De Casteljau's Solver" << endl; 

    dray::DeCasteljau:: solver; 

}

template <class Curve>
bool intersect(Array<Float> &res, Curve curveOne, Curve curveTwo, int maxIterations = 10, float threshold = 1e-3) {
    NormalizedImplicitLine line;
    FatLine fatLine; 
    Array<Float> intersections;
    Float t_min, t_max; 
    Float* resPtr = res.get_host_ptr();

    stack<Curve> paramStack;
    Curve clipCurve  = curveOne; 
    Curve otherCurve = curveTwo;

    size_t num_intersections = 0; 

    cout << "Starting the loopz" << endl; 

    while (true) {
        cout << "Current Iteration: " << maxIterations << endl;
        if (maxIterations == 0)
            return false;
        --maxIterations;

        cout << "Getting the line" << endl; 
        line    = normalized_implicit(clipCurve);

        cout << "Getting the fat line" << endl; 
        Point* temp = clipCurve.get_host_ptr(); 
        cout << "clipCurve: " << temp[0] << " " << temp[1] << endl; 
        fatLine = fat_line(line, clipCurve);
        
        // Find the points of intersection with between the second curve and 
        // the fat line. 
        cout << "Calling Intersection Points" << endl; 
        cout << "Fat Line: " << fatLine << endl; 
        cout << "Line-N: " << fatLine.line.N << endl; 
        cout << "Line-P_0: " << fatLine.line.P_0 << endl; 
        
        bool foundIntersections = intersection_points(otherCurve, fatLine, intersections); 
        cout << "Called intersection points" << endl; 
        Float* pointsPtr = intersections.get_host_ptr(); 
        if (!foundIntersections) {
            cout << "Did not find intersections." << endl; 
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

        cout << "Now checking the base case" << endl; 
        cout << "t_min: " << t_min << endl;
        cout << "t_max: " << t_max << endl; 

        // Base case
        if ((t_max - t_min) < threshold) {
            cout << "The base case is true!" << endl; 
            res.resize(res.size() + 1);

            cout << "The size is: " << res.size() << endl; 

            resPtr = res.get_host_ptr();
            
            cout << "TMax is: " << t_max << endl; 
            cout << "TMin is: " << t_min << endl; 
            

            resPtr[res.size() - 1] = (t_max + t_min) / 2; 

            cout << "Size: " << paramStack.size() << endl;
            if (paramStack.size() == 0) 
                ++num_intersections;
                break; 
            
            clipCurve  = paramStack.top(); 
            paramStack.pop();
            otherCurve = paramStack.top(); 
            paramStack.pop(); 
            continue; 
        }

        cout << "The t-range was not below the threshold" << endl; 

        // If we managed to clip less than 20% of the interval, there are probably two intersections. Split the curve into two 
        // and recursively call intersect() on the two halves.
        if ((t_max - t_min) > 0.80) {
            cout << "Not able to chop off enough of the curve" << endl; 
            Curve segmentOne; 
            Curve segmentTwo; 
            de_casteljau(segmentOne, segmentTwo, otherCurve, Float(0.5)); 
            paramStack.push(segmentTwo); 
            paramStack.push(clipCurve);  
            otherCurve = segmentOne;
            continue; 
        }  

        // Split the second curve at those points of intersection
        Curve segmentOne, segmentTwo, segmentThree; 
        de_casteljau(segmentOne, segmentTwo, otherCurve, pointsPtr[0]);   // Split at t_min 
        de_casteljau(segmentTwo, segmentThree, segmentTwo, pointsPtr[1]); // Split at t_max

        cout << "Switching the curves around" << endl; 
        cout << "Before switching, the sizes are: " << otherCurve.size() << " " << clipCurve.size() << endl;
        
        otherCurve = clipCurve;
        clipCurve  = segmentTwo; 

        cout << "After switching, the sizes are: " << otherCurve.size() << " " << clipCurve.size() << endl;
    }

    return num_intersections > 0; 
}

template <class Curve>
FatLine fat_line(NormalizedImplicitLine l, Curve curveOne) {
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
        Float p1_dist = l.dist(p1); // -18
        Float p2_dist = l.dist(p2); // 9
        minDist = (4. / 9) * std::min(std::min(p1_dist, p2_dist), Float(0));
        maxDist = (4. / 9) * std::max(std::max(p1_dist, p2_dist), Float(0));
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

    FatLine fatLine        = FatLine{};
    fatLine.upperBound     = maxDist;
    fatLine.lowerBound     = minDist;
    fatLine.line           = l; 
    return fatLine;
}

bool t_intersection(FatLine& line, Point& prevPoint, Point& currPoint, Float& t_min, Float& t_max, Float& t_0, Float& t_1, Float& t_interval) { 
    cout << "====" << endl; 
    cout << "Prev Point: " << prevPoint << endl; 
    cout << "Current Point: " << currPoint << endl;

    Float y_0  = line.line.dist(prevPoint); 
    Float y_1  = line.line.dist(currPoint); 
    t_1 += t_interval;  

    // Create a line segment by looking from the (i)th point to the (i-1)th point.
    // This starts at 'prevPoint' and goes to the currPoint. 
    Float slope = (y_1 - y_0) / (t_1 - t_0); 
    Float delta_t; 

    
    cout << "(t_0: " << t_0 << ", y_0: " << y_0 << ")" << endl; 
    cout << "(t_1: " << t_1 << ", y_1:" << y_1 << ")" << endl; 
    

    bool foundIntersection = false; 
    bool condOne = (y_0 <= line.upperBound && line.upperBound <= y_1);
    bool condTwo = (y_1 <= line.upperBound && line.upperBound <= y_0);
    
    cout << "Cond 1: " << condOne << " Cond 2: " << condTwo << endl; 

    if (condOne || condTwo) { 
        cout << "intersection with top." << endl; 
        delta_t = (line.upperBound - y_0) / slope; 
        t_min   = std::min(t_min, t_0 + delta_t);
        t_max   = std::max(t_max, t_0 + delta_t);
        foundIntersection = true; 
    } 
    condOne = (y_0 <= line.lowerBound && line.lowerBound <= y_1);
    condTwo = (y_1 <= line.lowerBound && line.lowerBound <= y_0);

    if (condOne || condTwo) { 
        cout << "intersection with bottom " << endl; 
        delta_t = (line.lowerBound - y_0) / slope;  
        t_min   = std::min(t_min, t_0 + delta_t);
        t_max   = std::max(t_max, t_0 + delta_t);
        foundIntersection = true; 
    } 

    // Update for next iteration
    prevPoint = currPoint; 
    t_0       = t_1;
    cout << "Found Intersection? " << foundIntersection << endl; 
    cout << "====" << endl; 
    return foundIntersection; 
}

// Takes a bezier curve and creates a normalized implicit line through it.
template <class Curve>
NormalizedImplicitLine normalized_implicit(Curve curve) {
    // First, create a line by looking at the first and last control point
    const Point* curvePtr = curve.get_host_ptr_const();  
    cout << "The curve has " << curve.size() << " points." << endl;
    Point firstControlPoint = curvePtr[0]; 
    size_t size = curve.size();
    Point lastControlPoint = curvePtr[size - 1];

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
            cout << "Looking at point (" << i << " , " << j << ")" << endl; 
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
template bool intersect(Array<Float> &res, Array<Point> curveOne, Array<Point> curveTwo, int maxIterations, float threshold);
template NormalizedImplicitLine normalized_implicit(Array<Point> curve);
template bool intersection_points(Array<Point> curve, FatLine line, Array<Float> &intersections);
    
}
}
