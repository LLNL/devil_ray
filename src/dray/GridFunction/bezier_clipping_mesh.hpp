#ifndef DRAY_BEZIER_CLIPPING_MESH_HPP
#define DRAY_BEZIER_CLIPPING_MESH_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/ray.hpp>
#include <dray/range.hpp>
#include <dray/GridFunction/bezier_clipping.hpp>

namespace dray 
{

namespace bezier_clipping 
{

template <uint p_order>  
using Mesh2D = MultiVec<Float, 2, 2, p_order>;

template <uint p_order>
using Mesh3D = MultiVec<Float, 2, 3, p_order>;
// We can think of this as an "n by m" matrix of control points
// (Meaning n rows, m columns). Mesh3D[i] is the ith row.

struct Plane {
    Vec3D n;
    Float o;

    friend std::ostream& operator<<(std::ostream& out, const Plane &p) {
        return out << "Normal: " << p.n << " Offset: " << p.o;
    } 
};

// Convert the ray to two intersecting planes.
void ray_to_planes(Plane &p1, Plane &p2, Ray ray); 

// Take the control points in the mesh and project them to 2D using the planes as the two
// axes.
template <uint p_order>
Mesh2D<p_order> project_mesh(const Plane &p1, const Plane &p2, Mesh3D<p_order> &mesh); 

// Get the lines Lu and Lv that represent the two directions of parametrization for the 
// projected mesh.
template <uint p_order>
void parametrized_directions(FatLine &l_u, FatLine &l_v, Mesh2D<p_order> mesh);

// Get the set of control points for the explicit mesh. 
// We also need to specify if we are creating an explicit mesh using l_u or l_v.
template <uint p_order>
Mesh2D<p_order> create_explicit_mesh(FatLine &l, Mesh2D<p_order>& projected_mesh, bool u_direction);

// Figure out the region [u_min, u_max] that could contain the intersection.
// Do this by finding the convex hull and seeing where it intersects the axis.
template <uint p_order>
bool region_of_interest(Mesh2D<p_order> explicit_mesh, Range &t_range, bool u_direction); 

// Find the intersection between a mesh and a ray.
template <uint p_order>
int intersect_mesh(Array<Float> &res, Ray ray, Mesh3D<p_order> mesh, Vec3D &intersection, int max_iter, Float threshold); 

}
}

#endif