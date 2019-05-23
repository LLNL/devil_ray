
// ===================================
// Class/Method List
//
// - Short descriptions of each class.
// - Summary of how the class interface should change or be replaced.
// - TODO collect methods that are not tethered to any class.
// ===================================================================
// Note: this is not a c++ file. The .cxx is for syntax highlighting.


// --------------------- //
// High-priority Changes //
// --------------------- //

/* @change Restrict the members of Ray to properties of rays. */
dray/ray.hpp:class Ray

/* @brief Program counters that disrupt the API and are hard to work with.
 * @change Architect a new modular program counter subsystem that doesn't get in the way. */
dray/utils/stats.hpp:struct _MultiReduceSum : public _MultiReduceSum<T,mult-1>
dray/utils/stats.hpp:struct _MultiReduceSum<T,0> {};
dray/utils/stats.hpp:struct HistogramSmall
dray/utils/stats.hpp:struct _NewtonSolveHistogram
dray/utils/stats.hpp:struct _NewtonSolveCounter
dray/utils/stats.hpp:////// struct Stats

/* @brief Templated abstraction for element transformations or element fields.
 *        - ElTransOp assumes ND hypercube with ND shape functions computed as
 *          a tensor product of 1D shape function, but makes no other assumptions
 *          about what the 1D shape functions are or how the dof values are stored.
 *        - A concrete implementation for ShapeOpType is BernsteinBasis.
 *        - A concrete implementation of CoeffIter is ElTransIter.
 *        - The separation is unneeded for simple element functions, but is necessary
 *          for compound element functions, like ElTransPairOp.
 *        - The purpose of ElTransPairOp is to fuse together two element functions
 *          and give them the appearance of a single element function having all the components.
 *        - The purpose of ElTransRayOp is to fuse together reference coordinates and the
 *          ray distance parameter, to deliver the difference between the corresponding
 *          world space coordinate and ray tip.
 *
 * @remark ElTransPairOp and ElTransRayOp are artificial interfaces. They were created so
 *         that the same implementation of Newton's method could be applied to 3 distinct systems,
 *         agnostic of the differences between them:
 *         1. Point-in-cell  == Element+WorldPoint;
 *         2. Ray-isosurface == Element+Ray+Field+Isovalue;
 *         3. Ray-boundary   == Element.Face+Ray
 *
 * @change We need to simplify the system by clearly defining object boundaries (encapuslation).
 *         Be able to do things like Mesh::get_element() and Field::get_element().
 *
 *         Short term: Encapsulate the needed element classes and provide good constructors.
 *                     Before changing everything, can hide the bad interface with factories like
 *                         ElTrans_XXX  make_face_element(Mesh, el_id, face_id);
 *                         ElTrans_YYY  fuse_ray_mesh_field_element(Ray, Mesh, Field, el_id);
 *         Long term:  Separate the solvers/intersectors so they are aware of the difference
 *                     between Mesh, Field, and Ray. Then get rid of ElTransPairOp and ElTransRayOp.
 *                     Also provide separate levels of granularity (L0/L1/L2).
 */
dray/el_trans.hpp:struct ElTransIter
dray/el_trans.hpp:struct ElTransOp : public ShapeOpType
dray/el_trans.hpp:struct ElTransBdryIter : public ElTransIter<T,PhysDim>
dray/el_trans.hpp:struct ElTransPairOp
dray/el_trans.hpp:struct ElTransRayOp : public ElTransOpType

/* @brief Houses the arrays for a Mesh or a Field.
 * @change Mesh or Field should keep the arrays themselves. Also should copy early and remove indirection. */
dray/el_trans.hpp:struct ElTransData


/* @brief Three different intersectors for Point-in-cell, Ray-isosurface, or Ray-boundary problems.
 *        They all call NewtonSolve::solve() to actually solve the system, so most of the code for
 *        each intersector is just pulling out device pointers, initializing structs, connecting pipes, and
 *        giving template arguments to the ElTransOp system.
 *
 * @change - Point-in-cell should not be considered an intersector,
 *           even though the implementation may be similar.
 *         - There should be separate levels of granularity (L0/L1/L2).
 *         - These methods should not have to do the initialization for elements, rather
 *           they should be passed an element (L0), or should call get_element() (L1/L2).
 *         - The intersector methods should implement or plug in different solvers, including Newton's method.
 */
dray/high_order_intersection.hpp:struct Intersector_PointVol
dray/high_order_intersection.hpp:struct Intersector_RayIsosurf
dray/high_order_intersection.hpp:  using RayType = struct { Vec<T,space_dim> dir; Vec<T,space_dim> orig; T isoval; };
dray/high_order_intersection.hpp:struct Intersector_RayBoundSurf
dray/high_order_intersection.hpp:  using RayType = struct { Vec<T,space_dim> dir; Vec<T,space_dim> orig; };

/* @brief BVH traversal, where the test is implicitly: Descend if Point \in AABB.
 * @change BVH should have a general traversal method that receives an AABB tester.
 *         Right now PointLocator and ray-AABB intersection are floating in different places.
 *         TODO need to figure out where in the API the bbox intersection methods should go. */
dray/point_location.hpp:class PointLocator

/* @brief Static method implementation of Newton's method, using ElTransOp-esque ::eval()
 *        method to get function values and partital derivatives.
 *
 * @change Short term: separate Newton's method into a NewtonStep and IterativeMethod.
 *         Long term: After the API is simplified, move the NewtonStep into the intersector functions.
 */
dray/newton_solver.hpp:struct NewtonSolve


/* @brief Ambient Occlusion.
 * @change Implementation using the new interface should be much simpler, no need for IntersectionContext. */
dray/ambient_occlusion.hpp:class AmbientOcclusion

/* @change These should no longer be necessary. */
dray/intersection_context.hpp:class IntersectionContext
dray/shading_context.hpp:class ShadingContext

/* @brief Pointer bundle to access aggregate of field data on the device.
 * @change make sure has const variants.
 *         separate into separate bundles for Mesh and for Field.
 *         access to bvh
 */
dray/high_order_shape.hpp:struct DeviceFieldData

/* @brief Volume rendering
 * @change Move the volume integration to a vis module and remove the duplicate code.
 */
dray/mfem_volume_integrator.hpp:class MFEMVolumeIntegrator
dray/high_order_shape.hpp:class MeshField

/* @brief Stub representing 4D BVH+Spanning tree.
 * @change Dissociate geometry from field. */
dray/high_order_shape.hpp:struct IsoBVH : public BVH



// --------------------- //
// Low-priority Changes  //
// --------------------- //

/* @change Make it a constexpr function */
dray/math.hpp:struct IntPow
dray/math.hpp:template <int32 b> struct IntPow<b,1> { enum { val = b }; };
dray/math.hpp:template <int32 b> struct IntPow<b,0> { enum { val = 1 }; };

/* @change 1. Seamless casting between Matrix and Vec<Vec<>>.
 *         2. Matrix view of the transpose. */
dray/vec.hpp:class Vec
dray/matrix.hpp:class Matrix

/* @brief Low-level polynomial evaluation, 1D.
 * @change Try methods that don't need aux_mem, e.g. repeated/hierarchical multiplication. Or just use pow(). */
dray/bernstein_basis.hpp:struct BernsteinBasis

/* @brief Old implementation of vis operations on native mfem data structures using mfem methods.
 * @change Can probably be deleted. Note that mesh and field are separate in this scheme. */
dray/mfem_grid_function.hpp:class MFEMGridFunction
dray/mfem_mesh.hpp:class MFEMMesh
dray/mfem_data_set.hpp:class MFEMDataSet

/* @brief Adapter from MFEM to DRay.
 * @change Implementation using new Element structure. Do the duplication here. */
dray/mfem2dray.hpp:// Import MFEM data from in-memory MFEM data structure.


// --------------------- //
// Other Wishlist Items  //
// --------------------- //

/* - Solver framework
 *   - E.g. iterative_solver(func():RefPoint-->NewRefPoint or Abort, init_refpt, ref_iter_tol, max_iter);
 *     This factors out the outer loop, yet still flexible for implicit Newton step or other methods */


// --------------------- //
// No Changes            //
// --------------------- //

/* @brief Axis-aligned bounding box (mutable). */
dray/aabb.hpp:class AABB
dray/range.hpp:class Range 

/* @brief Matt's Array class that is built on Umpire. */
dray/array.hpp:class Array
dray/array.hpp:template<typename t> class ArrayInternals;
dray/array_internals.hpp:class ArrayInternals : public ArrayInternalsBase
dray/array_internals_base.hpp:class ArrayInternalsBase
dray/array_registry.hpp:class ArrayInternalsBase;
dray/array_registry.hpp:class ArrayRegistry

/* @brief Implementation of color tables. */
dray/color_table.hpp:  struct ColorTableInternals;
dray/color_table.hpp:class ColorTable
dray/color_table.cpp:struct ColorControlPoint
dray/color_table.cpp:struct AlphaControlPoint
dray/color_table.cpp:struct ColorTableInternals

/* @brief Provider of meta-info about the program, most importantly the logo. */
dray/dray.hpp:class dray 

/* @brief System to build a BVH from AABBs; BVH tree data structure; intermediate arrays. */
dray/linear_bvh_builder.hpp:class LinearBVHBuilder
dray/linear_bvh_builder.hpp:struct BVH
dray/linear_bvh_builder.cpp:struct BVHData

/* @brief Functor to encode the ray-triangle intersection. */
dray/triangle_intersection.hpp:class TriLeafIntersector
dray/triangle_intersection.hpp:class Moller

/* @brief Global timing and data logging utilities. */
dray/utils/timer.hpp:class Timer 
dray/utils/data_logger.hpp:class DataLogger 
dray/utils/yaml_writer.hpp:class YamlWriter
dray/utils/yaml_writer.hpp:  struct Block

/* @brief DRayError. */
dray/error.hpp:class DRayError : public std::exception

/* @brief PNGEncoder. */
dray/utils/png_encoder.hpp:class PNGEncoder

/* @brief Triangle surface meshes to test graphics ideas. */
dray/triangle_mesh.hpp:class TriangleMesh

/* @brief Utilities for quickly computing binomial coefficients. Needed in BernsteinBasis. */
dray/binomial.hpp://class BinomTable
dray/binomial.hpp:struct BinomTable  //DEBUG
dray/binomial.hpp:struct BinomRow
dray/binomial.hpp:struct BinomT
dray/binomial.hpp:template <int32 n> struct BinomT<n,n> { enum { val = 1 }; };
dray/binomial.hpp:template <int32 n> struct BinomT<n,0> { enum { val = 1 }; };
dray/binomial.hpp:  struct BinomRowTInternal : public BinomRowTInternal<T,n,k-1>
dray/binomial.hpp:  struct BinomRowTInternal<T,n,0> { const T cell = static_cast<T>(1); };
dray/binomial.hpp:class BinomRowT

/* @brief Low-level polynomial evaluation, 1D, in the Power basis. Currently not being used. */
dray/power_basis.hpp:struct PowerBasis : public PowerBasis<T, RefDim-1>
dray/power_basis.hpp:struct PowerBasis<T, 1>

/* @brief ND tensor products of order-1 tensors. Not being used.
 * I think the intended use was to explicitly store shape function values. */
dray/simple_tensor.hpp:struct MultInPlace
dray/simple_tensor.hpp:struct MultInPlace<Vec<T,S>>
dray/simple_tensor.hpp:struct SimpleTensor   // This means single product of some number of vectors.

/* @brief Camera that can generate rays. */
dray/camera.hpp:class Camera

/* @brief Lighting/shading utilities, such as alpha blending. */
dray/shaders.hpp:struct PointLightSource
dray/shaders.hpp:class Shader
