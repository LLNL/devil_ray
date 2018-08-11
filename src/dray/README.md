# Devil Ray README : Some useful information

2018-08-10 Masado Ishii

## Spaces / Math

    Each element parameterizes a region of space by transforming reference coordinates to world coordinates. I'll use parametric coordinates (u,v,w) for element reference space, and world coordinates (x,y,z) for world space. Each element owns a transformation from its reference space to world space, \PHI: [0,1]^3 -> R^3, \PHI(u,v,w) = (x,y,z).

    For now we assume that \PHI is a trivariate polynomial (trivariate == 3 function of variables).

    A polynomial function can be represented in different bases. A basis is a special set of functions (we are talking about a linear basis in the vector space of polynomial functions). A given polynomial is a linear combination of the basis polynomials, which means each basis function has an associated coefficient, and then everything is added into a single function. From high school algebra, one usually uses the power basis, which looks like (1+10x+25x2). The special functions of the power basis are {1,x,x2,x3,...,xn}, and the coefficients are (1,10,25). There are other bases. Another is the Bernstein basis. Right now Devil Ray uses the trivariate Bernstein basis.

    I do not fully explain the Bernstein basis here, but here are some properties we are interested in:

    * Any polynomial evaluated in the unit cube will have a value within the convex hull of the set of coefficients relative to the Bernstein basis. This is a consequence of the "partition of unity" property of the Bernstein basis.
    * In the univariate Berntein basis, the 0th and nth coefficients are the values of the polynomial at the endpoints of the interval [0,1], respectively.
    * The trivariate Bernstein basis we use is a tensor product of three 1D Bernstein bases over different variables. A (p,p,p)-order polynomial has (p+1)3 coefficients, which can be arrayed in a cube lattice.
    * If you take a trivariate polynomial and restrict one of the variables to an endpoint (say you fix y=1), you get a bivariate polynomial whose coefficients in the Bernstein basis are a subset of the coefficients from the original polynomial. They are exactly the coefficients at the face of the cube lattice corresponding to the selected face of the reference space cube.

    Representing a high-order mesh: Assume a particular basis, such as the trivariate Bernstein basis of order (p,p,p), for some integer power p. A polynomial function of order p (or less) has a unique representation in this basis. To represent an element transformation in the Bernstein basis, one needs the coefficients corresponding to the basis functions. Because an element transformation has vector values, the coefficients are vectors. Equivalently, each spatial component can be considered a separate scalar trivariate polynomial.

    Aside from the element transformation, any scalar field defined on an element can be represented using another set of coefficients. The coefficients for a scalar field will be scalars. (The polynomial basis for a scalar field need not be the same order, or even the same basis, as that used for the element transformation).

    In the finite element method, the coefficients of basis polynomials are also called "degrees of freedom." Sometimes (e.g., in H1 continuous spaces), adjacent elements share degrees of freedom along their edges.

    Thus, to represent a mesh, we have to store two arrays:

    * a set of degrees of freedom (array of Vec3f); and
    * a set of local lookup tables (array of int32).

    Each local lookup table is a map from element coefficient positions, to global degree-of-freedom positions.

    MFEM uses a representation like this. Devil Ray essentially uses the same representation, but note that I have enforced different assumptions about the order of things in DRay.


## Some algorithms and explanation

Algorithm: _Point location_

Input: Geometry (dray::MeshField), World points (dray::Array<dray::Vec3f>)

Output: Element id and reference point for each input world point. (dray::Array<dray::int32>, dray::Array<dray::Vec3f>)

1. RAJA::forall<>({each point P})

    1. BVH traversal. Before we can worry about reference coordinates, we need an element. BVH traversal gives -> list of candidates.

    2. foreach(candidate)

        1. Put together an ElTrans object that specifies the polynomial order and element coefficients

            1. The polynomial order is specified somewhere in the mesh (I have assumed that all elements in the mesh have the same order).

            2. The polynomial evaluator component of ElTrans is BernsteinBasis. BernsteinBasis requires a pointer to a read/writable sub-array of main memory in order to evaluate the 1D Bernstein basis polynomials.

            3. The coefficient-iterator part of ElTrans requires pointers to the coefficient value array, coefficient lookup array, and the element index of the current candidate.

            // Result: The ElTrans object now has enough information to transform any reference point to a world space point.

        2. Solve the equation \PHI(u,v,w) = P, using the Newton-Raphson method.

            1. Set an initial guess. (For now, it is the middle, (.5,.5,.5).)

            2. For each iteration of Newton's method

                // *Note: Here I have used Q and U, but in the code they are y and x.*

                1. Evaluate Q = \PHI(u,v,w) and its partial derivatives at (u,v,w).
                2. Compute \delta Q = (P - Q). Compare Q with target P. If close enough, exit.
                3. Otherwise, perform the Newton step.
                    1. Construct a matrix J (the jacobian of \PHI) from the partial derivatives of \PHI that we evaluated earlier.
                    2. Compute the reference increment by solving J \delta U = \delta Q for \delta U. This uses either matrix inversion or LU_solve().
                    3. In case of a singularity, exit.  //TODO might be good to add a return type of error.
                    4. Otherwise, add the increment \delta U to (u,v,w).
                    5. If the increment \delta U barely changed, exit because we either have arrived or never will.

             // Result: A new reference point (u',v',w').

        3. If the Newton-Raphson was successful then use the current candidate (a point shall be contained in at most one element).
        //TODO this test is "!=NotConverged," but maybe it should be "==ConvergePhys"

        4. Otherwise, continue with the next candidate.

    // Result: Either (el_id, ref_pt), or (-1, __).






