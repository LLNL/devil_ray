#include <dray/GridFunction/bezier_clipping_mesh.hpp>
#include <dray/range.hpp>
#include <stack> 
#include <math.h>

namespace dray 
{

namespace bezier_clipping 
{
void ray_to_planes(Plane &p1, Plane &p2, Ray ray)
{
    // Define our orignal basis vectors;
    Vec3D x_basis = {1, 0, 0};
    Vec3D y_basis = {0, 1, 0};
    Vec3D z_basis = {0, 0, 1};
    ray.m_dir.normalize(); 

    // We need two rotations to get to our ray
    // one about the +z axis, and one about the +y axis.

    // First we rotate about y (by phi), then we rotate about z (by theta);
    Float phi = asin(-ray.m_dir[2]);
    Float theta;
    if (abs(1 - ray.m_dir[0] / cos(phi)) < .00001) {
        theta = 0;
    } else if (abs(-1 - ray.m_dir[0] / cos(phi)) < .00001) { 
        theta = M_PI;
    } else {
        theta = acos(ray.m_dir[0] / cos(phi));
    }
    
    Vec3D y_rot_x = Vec3D{cos(phi), 0, sin(phi)}; 
    Vec3D y_rot_y = Vec3D{0, 1, 0}; 
    Vec3D y_rot_z = Vec3D{-sin(phi), 0, cos(phi)};

    Vec3D z_rot_x = Vec3D{cos(theta), -sin(theta), 0}; 
    Vec3D z_rot_y = Vec3D{sin(theta), cos(theta), 0}; 
    Vec3D z_rot_z = Vec3D{0, 0, 1};

    x_basis = {dot(y_rot_x, x_basis), dot(y_rot_y, x_basis), dot(y_rot_z, x_basis)};
    y_basis = {dot(y_rot_x, y_basis), dot(y_rot_y, y_basis), dot(y_rot_z, y_basis)};
    z_basis = {dot(y_rot_x, z_basis), dot(y_rot_y, z_basis), dot(y_rot_z, z_basis)};
    
    x_basis = {dot(z_rot_x, x_basis), dot(z_rot_y, x_basis), dot(z_rot_z, x_basis)};
    y_basis = {dot(z_rot_x, y_basis), dot(z_rot_y, y_basis), dot(z_rot_z, y_basis)};
    z_basis = {dot(z_rot_x, z_basis), dot(z_rot_y, z_basis), dot(z_rot_z, z_basis)};

    p1.n = y_basis; 
    p2.n = z_basis;
    p1.o = -1 * dot(p1.n, ray.m_orig);
    p2.o = -1 * dot(p2.n, ray.m_orig); 
}

template <uint m>
Mesh2D<m> project_mesh(const Plane &p1, const Plane &p2, Mesh3D<m> &mesh)
{
    Mesh2D<m> projected_mesh; 
    
    for (int i = 0; i <= m; ++i) {
        int j = 0;
        for (auto &point : projected_mesh[i].components()) {
            point = Vec2D{dot(p1.n, mesh[i][j]) + p1.o, dot(p2.n, mesh[i][j]) + p2.o};
            ++j;
        }
    }
    
    return projected_mesh;
}

template <uint p> 
void parametrized_directions(FatLine &l_u, FatLine &l_v, Mesh2D<p> mesh)
{
    Vec2D dir_u = (mesh[p][0] - mesh[0][0]) + (mesh[p][p] - mesh[0][p]);
    Vec2D dir_v = (mesh[0][p] - mesh[0][0]) + (mesh[p][p] - mesh[p][0]);
    l_u.N = {-1 * dir_u[1], dir_u[0]};
    l_v.N = {-1 * dir_v[1], dir_v[0]};
    l_u.P_0 = Vec2D{0., 0.};
    l_v.P_0 = Vec2D{0., 0.};
    l_u.N.normalize(); 
    l_v.N.normalize();
}

template <uint p_order>
Mesh2D<p_order> create_explicit_mesh(FatLine &l, Mesh2D<p_order> &projected_mesh, bool u_direction)
{
    Mesh2D<p_order> res;

    for (int i = 0; i <= p_order; ++i) {
        int j = 0; 
        for (auto &point : res[i].components()) {
            Float t_val, dist;
            if (u_direction) {
                t_val = Float(i) / p_order;
            } else {
                // Equally spaced points on [0, 1].
                t_val = Float(j) / p_order; 
            }
            
            point = Vec2D{t_val, l.dist(projected_mesh[i][j])};
            ++j; 
        }
    }

    return res;
}

template <uint m>
bool region_of_interest(Mesh2D<m> explicit_mesh, Range &t_range, bool u_direction)
{    
    t_range.reset(); 

    Float min_dist, max_dist; 
    Float min_dist_prev, max_dist_prev;
    bool found_intersection = false;

    // This is just a special case of the curve-curve algorithm where the upper 
    // and lower bounds are 0.
    FatLine l = FatLine{Vec2D{0, 0}, Vec2D{0, 0}, Range::zero()}; 

    // Use direction as input so that we can look at all of the explicit points wih the same t
    // value at each iteration.
    
    // Distance between each control point along 't' axis is calculated based on 
    // number of control points we have along the parametrization direction we're
    // looking at.
    Float t_interval = 1. / m;

    for (int i = 0; i <= m; ++i) { 
        min_dist_prev = min_dist;
        max_dist_prev = max_dist;

        if (u_direction) { 
            min_dist = explicit_mesh[i][0][1];
            max_dist = explicit_mesh[i][0][1];
        } else {
            min_dist = explicit_mesh[0][i][1];
            max_dist = explicit_mesh[0][i][1];
        }
        
        for (int j = 1; j <= m; ++j) { 
            if (u_direction) {
                min_dist = std::min(min_dist, explicit_mesh[i][j][1]); 
                max_dist = std::max(max_dist, explicit_mesh[i][j][1]);
            } else { 
                min_dist = std::min(min_dist, explicit_mesh[j][i][1]); 
                max_dist = std::max(max_dist, explicit_mesh[j][i][1]);
            }
        }

        if (i == 0) {
            if (min_dist < 0 && max_dist > 0)
                t_range.update(0., 0.);

            // No previous point in this case.
            continue;
        } 
        
        found_intersection = t_intersection(l, min_dist_prev, min_dist, t_range, 
                                            (i - 1) * t_interval, i * t_interval) || 
                                            found_intersection;
        found_intersection = t_intersection(l, max_dist_prev, max_dist, t_range, 
                                            (i - 1) * t_interval, i * t_interval) || 
                                            found_intersection;
        
    }

    if (min_dist < 0. && max_dist > 0) {
        t_range.update(t_range.min(), 1.);
        if (t_range.min() > 1.)
            t_range.update(1., t_range.max());
        found_intersection = true;
    }

    return found_intersection;
}

template <uint p_order>
struct MeshStruct {
    Range u_d; 
    Range v_d;
    Mesh2D<p_order> mesh;
};

template <uint m>
int intersect_mesh(Array<Float> &res, Ray ray, Mesh3D<m> mesh, Vec3D &intersection,
              int max_iter = 10, Float threshold = 0.001)
{
    using Mesh2D = Mesh2D<m>;
    using MeshStruct = MeshStruct<m>;

    Plane p1, p2;
    FatLine l_u, l_v;
    int num_intersections = 0;

    ray_to_planes(p1, p2, ray);
    Mesh2D projected_mesh = project_mesh(p1, p2, mesh);
    parametrized_directions(l_u, l_v, projected_mesh);
    
    bool u_direction = true;
    MeshStruct projected_mesh_data = MeshStruct{Range::ref_universe(), Range::ref_universe(), projected_mesh}; 
    stack<MeshStruct> param_stack;

    while (true) {
        if (--max_iter < 0)
            return 0;

        Mesh2D explicit_mesh;
        Range t_range;
    
        if (u_direction)
            explicit_mesh = create_explicit_mesh(l_u, projected_mesh, true);
        else 
            explicit_mesh = create_explicit_mesh(l_v, projected_mesh, false);

        bool found_intersection = region_of_interest(explicit_mesh, t_range, u_direction);

        // No intersection found. If there are other parts of the mesh we have no looked at, continue.
        if (!found_intersection) {
            if (param_stack.size() == 0) 
                break; 

            projected_mesh_data = param_stack.top();
            param_stack.pop(); 
            continue;
        }

        Float delta_u = projected_mesh_data.u_d.max() - projected_mesh_data.u_d.min();
        Float delta_v = projected_mesh_data.v_d.max() - projected_mesh_data.v_d.min();

        // If we managed to clip less than 20% of the interval, there are probably two intersections.
        // Split the curve into two and recursively call intersect() on the two halves.
        if (u_direction && delta_u > 0.80) { 
            // TODO

            continue;
        } else if (!u_direction && delta_v > 0.80) { 
            // TODO

            continue;
        }

        // Now do the actual clipping (TODO)

        // Base Case 
        if (delta_u < threshold && delta_v < threshold) { 
            ++num_intersections;
            res.resize(res.size() + 2);

            Float *resPtr = res.get_host_ptr();    
            resPtr[res.size() - 1] = projected_mesh_data.u_d.center();
            resPtr[res.size() - 2] = projected_mesh_data.v_d.center();

            if (param_stack.size() == 0) 
                break; 
            
            projected_mesh_data = param_stack.top(); 
            param_stack.pop();
            continue; 
        }

        // Check if we no longer need to clip in one of the two directions.
        if (delta_u < threshold) {
            u_direction = false;
        } else if (delta_v < threshold) { 
            u_direction = true;
        } else { 
            u_direction = !u_direction;
        }
    }

    return num_intersections;
}

// ==============================
// === EXPLICIT INSTANTIATION ===
// ==============================
template Mesh2D<1u> project_mesh(const Plane &p1, const Plane &p2, Mesh3D<1u> &mesh);
template Mesh2D<2u> project_mesh(const Plane &p1, const Plane &p2, Mesh3D<2u> &mesh);

template void parametrized_directions(FatLine &l_u, FatLine &l_v, Mesh2D<1u> mesh);
template void parametrized_directions(FatLine &l_u, FatLine &l_v, Mesh2D<2u> mesh);

template Mesh2D<1u> create_explicit_mesh(FatLine &l, Mesh2D<1u>& projected_mesh, bool u_direction);
template Mesh2D<2u> create_explicit_mesh(FatLine &l, Mesh2D<2u>& projected_mesh, bool u_direction);

template bool region_of_interest(Mesh2D<1u> explicit_mesh, Range &t_range, bool u_direction);
template bool region_of_interest(Mesh2D<2u> explicit_mesh, Range &t_range, bool u_direction);

template int intersect_mesh(Array<Float> &res, Ray ray, Mesh3D<1u> mesh, Vec3D &intersection,
                            int max_iter, Float threshold);
}
}
