// *************************************************************************
// ******   First scratch of my basic lib for my computational need  *******
// *************************************************************************
#[allow(dead_code)]
pub mod geometry {
    use core::f64;
    use std::f64::EPSILON;
    // Implementation of a Point3d structure
    // bound to Vector3d structure
    // for standard operator processing.
    use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};
    use crate::display_pipe_line::redering_object::Vertex; 

    #[allow(non_snake_case)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Point3d {
        pub X: f64,
        pub Y: f64,
        pub Z: f64,
    }

    #[allow(non_snake_case)]
    impl Point3d {
        ///  Create a 3d point.
        ///  # Arguments
        ///  (x:f64, y:f64, z:f64)
        ///  # Returns
        ///  a new Point3d from x,y,z values.
        pub fn new(x: f64, y: f64, z: f64) -> Self {
            Self { X: x, Y: y, Z: z }
        }

        /// Test if a point is on a plane.
        pub fn is_on_plane(&self, plane: &[Point3d; 4]) -> bool {
            let normal =
                Vector3d::cross_product(&((*plane)[1] - (*plane)[0]), &((*plane)[3] - (*plane)[0]));
            let d =
                -(normal.X * (*plane)[0].X + normal.Y * (*plane)[0].Y + normal.Z * (*plane)[0].Z);
            if ((normal.X * self.X + normal.Y * self.Y + normal.Z * self.Z) + d).abs() < 1e-5 {
                true
            } else {
                false
            }
        }
    }

    // Implementation of + and - operator for Point3d.
    impl Add for Point3d {
        type Output = Self; // Specify the result type of the addition
        fn add(self, other: Self) -> Self {
            Self {
                X: self.X + other.X,
                Y: self.Y + other.Y,
                Z: self.Z + other.Z,
            }
        }
    }

    impl Sub for Point3d {
        type Output = Vector3d; // Specify the result type of the addition
        fn sub(self, other: Self) -> Vector3d {
            Vector3d::new(self.X - other.X, self.Y - other.Y, self.Z - other.Z)
        }
    }
    impl Mul<f64> for Point3d {
        type Output = Point3d;
        fn mul(self, scalar: f64) -> Point3d {
            Point3d {
                X: self.X * scalar,
                Y: self.Y * scalar,
                Z: self.Z * scalar,
            }
        }
    }

    impl Add<Vector3d> for Point3d {
        type Output = Point3d;
        fn add(self, vector: Vector3d) -> Point3d {
            Point3d {
                X: self.X + vector.X,
                Y: self.Y + vector.Y,
                Z: self.Z + vector.Z,
            }
        }
    }

    impl AddAssign<Vector3d> for Point3d {
        fn add_assign(&mut self, other: Vector3d) {
            self.X = self.X + other.X;
            self.Y = self.Y + other.Y;
            self.Z = self.Z + other.Z;
        }
    }

    impl MulAssign<f64> for Point3d {
        fn mul_assign(&mut self, scalar: f64) {
            self.X *= scalar;
            self.Y *= scalar;
            self.Z *= scalar;
        }
    }
    // the following implementation for Point3d is when Point3d is use as ambigius 
    // representation of a Vector3d in order to avoid runtime penalty.
    // - using a Vertex will full fill the same purpose in a more idiomatic way.
    impl Point3d {
        /// Compute the magnitude of the point vector
        pub fn magnitude(&self) -> f64 {
            (self.X * self.X + self.Y * self.Y + self.Z * self.Z).sqrt()
        }

        /// Normalize the the point as a vector
        /// (equivalent to unitize_b for Vector3d)
        /// this remove embiguity when point 3d is use as vector 
        /// ( to avoid sqrt penalty on magnetide creation when using Vector3d )
        /// - it's recomended to use Vertex for that.
        pub fn normalize(&self) -> Self {
            let mag = self.magnitude();
            if mag > std::f64::EPSILON {
                Point3d::new(self.X / mag, self.Y / mag, self.Z / mag)
            } else {
                Point3d::new(0.0, 0.0, 0.0)
            }
        }

        /// Compute the cross product of two vectors
        pub fn cross(&self, other: &Point3d) -> Point3d {
            Point3d {
                X: self.Y * other.Z - self.Z * other.Y,
                Y: self.Z * other.X - self.X * other.Z,
                Z: self.X * other.Y - self.Y * other.X,
            }
        }
        /// Convert a Point3d to Vertex.
        pub fn to_vertex(&self)->Vertex{
            Vertex::new(self.X ,self.Y, self.Z)
        }
    }

    // Vector 3d definition.
    #[allow(non_snake_case)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Vector3d {
        X: f64,
        Y: f64,
        Z: f64,
        Length: f64,
    }

    #[allow(non_snake_case)]
    impl Vector3d {
        ///  Create a 3d vector.
        ///  # Arguments
        ///  (x:f64, y:f64, z:f64)
        ///  # Returns
        ///  a new Vector 3d from x,y,z values.
        ///  /////////////////////////////////////////////////////////
        ///   the vector Length is automatically computed at vector
        ///   creation or modification of one of it's X,Y,Z components
        ///  //////////////////////////////////////////////////////////
        pub fn new(x: f64, y: f64, z: f64) -> Self {
            Self {
                X: x,
                Y: y,
                Z: z,
                // making a copy of an f64 is faster than
                // moving and dereferencing a pointer on a f64...
                Length: Vector3d::compute_length(x, y, z),
            }
        }
        pub fn set_X(&mut self, new_x_value: f64) {
            self.X = new_x_value;
            self.update_length();
        }
        pub fn set_Y(&mut self, new_y_value: f64) {
            self.Y = new_y_value;
            self.update_length();
        }
        pub fn set_Z(&mut self, new_z_value: f64) {
            self.Z = new_z_value;
            self.update_length();
        }
        pub fn get_X(&self) -> f64 {
            self.X
        }
        pub fn get_Y(&self) -> f64 {
            self.Y
        }
        pub fn get_Z(&self) -> f64 {
            self.Z
        }

        /// Return the read only length
        pub fn Length(&self) -> f64 {
            self.Length
        }

        /// Compute the vector length.
        fn update_length(&mut self) {
            self.Length = (self.X * self.X + self.Y * self.Y + self.Z * self.Z).sqrt();
        }
        /// Convert to a Vertex
        pub fn to_vertex(&self)->Vertex{
            Vertex::new(self.X,self.Y,self.Z)
        }
        /// static way to compute vector length.
        /// # Arguments
        /// x:f 64, y:f 64, z:f 64
        /// # Returns
        /// return a f 64 length distance.
        pub fn compute_length(x: f64, y: f64, z: f64) -> f64 {
            (x * x + y * y + z * z).sqrt()
        }

        pub fn compute_length_byref(x: &f64, y: &f64, z: &f64) -> f64 {
            ((*x) * (*x) + (*y) * (*y) + (*z) * (*z)).sqrt()
        }

        /// Represent the rate of change between two points in time
        /// where the magnitude of the result is the speed expressed
        /// in unit of time/space relationship of the input.
        pub fn compute_velocity(
            initial_position: &Point3d,
            final_position: &Point3d,
            initial_time: f64,
            final_time: f64,
        ) -> Vector3d {
            let delta_position = (*final_position) - (*initial_position);
            let delta_time = final_time - initial_time;
            delta_position * (1.0 / delta_time)
        }

        /// Compute acceleration vector given
        /// by two velocity vectors and a time interval.
        pub fn compute_acceleration(
            initial_velocity: &Vector3d,
            final_velocity: &Vector3d,
            initial_time: f64,
            final_time: f64,
        ) -> Vector3d {
            let delta_velocity = (*final_velocity) - (*initial_velocity);
            let delta_time = final_time - initial_time;
            delta_velocity * (1.0 / delta_time)
        }

        // Vector3d Cross Product.
        pub fn cross_product(vector_a: &Vector3d, vector_b: &Vector3d) -> Self {
            Vector3d::new(
                (*vector_a).Y * (*vector_b).Z - (*vector_a).Z * (*vector_b).Y,
                (*vector_a).Z * (*vector_b).X - (*vector_a).X * (*vector_b).Z,
                (*vector_a).X * (*vector_b).Y - (*vector_a).Y * (*vector_b).X,
            )
        }

        // Unitize the Vector3d.
        pub fn unitize(&mut self) {
            self.X /= self.Length;
            self.Y /= self.Length;
            self.Z /= self.Length;
            self.update_length();
        }

        pub fn unitize_b(&self) -> Vector3d {
            if self.Length > std::f64::EPSILON {
                // set very tinny vector to zero.
                Vector3d::new(
                    self.X / self.Length,
                    self.Y / self.Length,
                    self.Z / self.Length,
                )
            } else {
                Vector3d::new(0.0, 0.0, 0.0)
            }
        }


        /// Test if a vector point to the direction of an other vector.
        /// # Arguments
        /// - ref &self,
        /// - other_vector:Vector3d (other vector to compare with),
        /// - threshold :f64
        /// (threshold should be closer to one for getting more precision. like: 0.99991)
        /// #Returns
        /// - true if looking in same direction or false if looking in other direction
        ///   (always from threshold value).
        pub fn is_same_direction(&self, other_vector: &Vector3d, threshold: f64) -> bool {
            if (*self).unitize_b() * (*other_vector).unitize_b() >= threshold {
                true
            } else {
                false
            }
        }

        /// return the angle between two vectors
        pub fn compute_angle(vector_a: &Vector3d, vector_b: &Vector3d) -> f64 {
            f64::acos(
                ((*vector_a) * (*vector_b))
                    / (f64::sqrt(
                        (*vector_a).X * (*vector_a).X
                            + (*vector_a).Y * (*vector_a).Y
                            + (*vector_a).Z * (*vector_a).Z,
                    ) * f64::sqrt(
                        (*vector_b).X * (*vector_b).X
                            + (*vector_b).Y * (*vector_b).Y
                            + (*vector_b).Z * (*vector_b).Z,
                    )),
            )
        }

        /// Test if two vectors sit on a same 3d plane.
        pub fn are_coplanar_a(vector_a: &Vector3d, vector_b: &Vector3d) -> bool {
            let vector_c = (*vector_b) - (*vector_a);
            if (Vector3d::cross_product(vector_a, vector_b) * (vector_c)).abs() <= 1e-5 {
                true
            } else {
                false
            }
        }

        /// Test if tree vectors are coplanar with the scalar triple product.
        /// (if the volume of the AxB (cross product) * C == 0 they are coplanar)
        /// notes: two vectors addition make a third vector.
        pub fn are_coplanar_b(
            vector_a: &Vector3d,
            vector_b: &Vector3d,
            vector_c: &Vector3d,
        ) -> bool {
            if (Vector3d::cross_product(vector_a, vector_b) * (*vector_c)).abs() <= 1e-5 {
                true
            } else {
                false
            }
        }
        
        /// Test if two vectors are perpendicular.
        pub fn are_perpandicular(vector_a: &Vector3d, vector_b: &Vector3d) -> bool {
            if ((*vector_a) * (*vector_b)).abs() <= EPSILON {
                true
            } else {
                false
            }
        }

        /// Rotate a vector around an axis using Rodrigues' rotation formula.
        /// input angle is in radians.
        pub fn rotate_around_axis(self, axis: &Vector3d, angle: f64) -> Vector3d {
            let unit_axis = (*axis).unitize_b();
            let cos_theta = angle.cos();
            let sin_theta = angle.sin();

            let v_parallel = unit_axis * (self * unit_axis); // Projection of v onto axis
            let v_perpendicular = self - v_parallel; // Perpendicular component of v

            // Rotated perpendicular component
            let v_rotated_perpendicular = Vector3d::cross_product(&self, &unit_axis) * sin_theta;

            (v_perpendicular * cos_theta) + v_rotated_perpendicular + v_parallel
        }

        /// Project a vector on an infinite plane.
        /// # Arguments
        ///   takes plane as an array of two coplanar vectors from a same origin 3d point
        ///   defining the edge of the parallelepiped.
        /// # Returns
        ///  - an Option<Vector3d>
        ///  - the projected Vector on success or  None on failure.
        /// ! for a valid result:
        ///   - vectors must describes two edges of the plane
        ///     starting from the same origin
        pub fn project_on_infinite_plane(&self, plane: &[Vector3d; 2]) -> Option<Vector3d> {
            if Vector3d::are_coplanar_a(&((*plane)[0]), &((*plane)[1])) {
                let normal = Vector3d::cross_product(&(*plane)[0], &(*plane)[1]).unitize_b();
                Some((*self) - (normal * ((*self) * normal)))
            } else {
                None
            }
        }
    }
    impl Neg for Vector3d {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Self {
                X: -self.X,
                Y: -self.Y,
                Z: -self.Z,
                Length: self.Length,
            }
        }
    }
    impl Mul for Vector3d {
        type Output = f64;
        fn mul(self, vector: Vector3d) -> f64 {
            self.X * vector.X + self.Y * vector.Y + self.Z * vector.Z
        }
    }

    impl Sub for Vector3d {
        type Output = Self; // Specify the result type of the addition
        fn sub(self, other: Self) -> Self {
            Vector3d::new(self.X - other.X, self.Y - other.Y, self.Z - other.Z)
        }
    }

    impl Add for Vector3d {
        type Output = Self;
        fn add(self, vector: Vector3d) -> Self {
            Vector3d::new(self.X + vector.X, self.Y + vector.Y, self.Z + vector.Z)
        }
    }

    impl Mul<f64> for Vector3d {
        type Output = Vector3d;
        fn mul(self, scalar: f64) -> Self {
            let v_x = self.X * scalar;
            let v_y = self.Y * scalar;
            let v_z = self.Z * scalar;
            Vector3d::new(v_x, v_y, v_z)
        }
    }

    impl Div<f64> for Vector3d {
        type Output = Vector3d;
        fn div(self, scalar: f64) -> Self {
            let v_x = self.X / scalar;
            let v_y = self.Y / scalar;
            let v_z = self.Z / scalar;
            Vector3d::new(v_x, v_y, v_z)
        }
    }

    impl MulAssign<f64> for Vector3d {
        fn mul_assign(&mut self, scalar: f64) {
            self.X *= scalar;
            self.Y *= scalar;
            self.Z *= scalar;
            self.update_length();
        }
    }

    pub struct CPlane {
        pub origin: Point3d,
        pub normal: Vector3d,
        pub u: Vector3d, // Local X axis on the plane
        pub v: Vector3d, // Local Y axis on the plane
    }

    impl CPlane {
        /// Constructs a plane from an origin and a normal vector
        pub fn new(origin: &Point3d, normal: &Vector3d) -> Self {
            let normalized_normal = (*normal).unitize_b();

            // Find a vector that is not parallel to the normal
            let mut arbitrary_vector = Vector3d::new(1.0, 0.0, 0.0);
            if (*normal).X.abs() > 0.99 {
                arbitrary_vector = Vector3d::new(0.0, 1.0, 0.0);
            }

            // Compute two orthogonal vectors on the plane using the cross product
            let u = Vector3d::cross_product(&normalized_normal, &arbitrary_vector).unitize_b();
            let v = Vector3d::cross_product(&normalized_normal, &u).unitize_b();

            Self {
                origin: Point3d::new((*origin).X, (*origin).Y, (*origin).Z),
                normal: normalized_normal,
                u,
                v,
            }
        }
        /// Construct a CPlane from origin point and a normal vector
        /// oriented on x_axis direction.
        /// # Arguments
        /// - a &Point3d origin,
        /// - a &Vector3d x_axis direction,
        /// - a &Vector3d normal of the plane.
        /// # Returns
        /// return always a plane oriented with the x_axis input Vector3d but locked
        /// at 90 deg from the plane normal Vector3d (Plane orthogonalization process).
        /// (the normal vector is the reference input for orthogonalization)
        /// normal vector influence light so might be considered first.
        pub fn new_normal_x_oriented(
            origin: &Point3d,
            x_axis: &Vector3d,
            normal: &Vector3d,
        ) -> Self {
            // normalize the normal.
            let normalized_normal = normal.unitize_b();
            // Define local Y 'v' from normal at 90 deg (orthogonalization).
            let v = Vector3d::cross_product(&normalized_normal, &x_axis).unitize_b();
            // Define local X 'u' always relative to 90deg from (y,z) plane.
            let u = -Vector3d::cross_product(&v, &normalized_normal).unitize_b();
            Self {
                origin: *origin,
                normal: normalized_normal,
                u,
                v,
            }
        }

        /// Construct a CPlane from 3 points.
        /// # Arguments
        /// - &Point3d origin of the plane.
        /// - &Point3d pt_x aligned on the x_axis direction.
        /// - &Point3d pt_y oriented on the y_axis direction.
        /// # Returns
        /// return a CPlane with the normal orthogonalized from the three points.
        pub fn new_origin_x_aligned_y_oriented(
            origin: &Point3d,
            pt_x: &Point3d,
            pt_y: &Point3d,
        ) -> Self {
            // Define (u,v) Vector3d(s).
            let x_axis = ((*pt_x) - (*origin)).unitize_b();
            let mut y_axis = ((*pt_y) - (*origin)).unitize_b();
            // compute the normal.
            let normal = Vector3d::cross_product(&x_axis, &y_axis).unitize_b();
            // make sure that v is orthogonal to u.
            y_axis = Vector3d::cross_product(&x_axis, &normal).unitize_b();
            Self {
                origin: *origin,
                normal,
                u: x_axis,
                v: y_axis,
            }
        }

        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        pub fn point_on_plane_uv(&self, u: f64, v: f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * u + self.v.X * v,
                Y: self.origin.Y + self.u.Y * u + self.v.Y * v,
                Z: self.origin.Z + self.u.Z * u + self.v.Z * v,
            }
        }

        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        /// Also offsets the point along the plane's normal by z value.
        pub fn point_on_plane(&self, x: f64, y: f64, z: f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * x + self.v.X * y + self.normal.X * z,
                Y: self.origin.Y + self.u.Y * x + self.v.Y * y + self.normal.Y * z,
                Z: self.origin.Z + self.u.Z * x + self.v.Z * y + self.normal.Z * z,
            }
        }
        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        pub fn point_on_plane_uv_ref(&self, u: &f64, v: &f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * (*u) + self.v.X * (*v),
                Y: self.origin.Y + self.u.Y * (*u) + self.v.Y * (*v),
                Z: self.origin.Z + self.u.Z * (*u) + self.v.Z * (*v),
            }
        }

        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        /// Also offsets the point along the plane's normal by z value.
        pub fn point_on_plane_ref(&self, x: &f64, y: &f64, z: &f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * (*x) + self.v.X * (*y) + self.normal.X * (*z),
                Y: self.origin.Y + self.u.Y * (*x) + self.v.Y * (*y) + self.normal.Y * (*z),
                Z: self.origin.Z + self.u.Z * (*x) + self.v.Z * (*y) + self.normal.Z * (*z),
            }
        }
    }

    pub struct NurbsCurve {
        pub control_points: Vec<Point3d>,
        pub degree: usize,
        pub knots: Vec<f64>,
        pub weights: Vec<f64>,
    }

    impl NurbsCurve {
        /// Constructor for a NURBS curve
        pub fn new(control_points: Vec<Point3d>, degree: usize) -> Self {
            let n = control_points.len(); // Number of control points
            let num_knots = n + degree + 1; // Number of knots

            // Generate a clamped knot vector
            let mut knots = vec![0.0; num_knots];
            let interior_knots = n - degree; // Number of non-clamped (interior) knots

            // Fill the clamped start and end
            for i in 0..=degree {
                knots[i] = 0.0; // Clamped at the start
                knots[num_knots - 1 - i] = 1.0; // Clamped at the end
            }

            // Fill the interior knots with uniform spacing
            for i in 1..interior_knots {
                knots[degree + i] = i as f64 / interior_knots as f64;
            }

            // Default weights (all 1.0)
            let weights = vec![1.0; n];

            // Construct the NurbsCurve
            NurbsCurve {
                control_points,
                degree,
                knots,
                weights,
            }
        }
        /// Compute the curvature of the NURBS curve at parameter t
        pub fn curvature(&self, t: f64) -> f64 {
            let first_derivative = self.numerical_first_derivative(t, 1e-5);
            let second_derivative = self.numerical_second_derivative(t, 1e-5);
            let cross_product = first_derivative.cross(&second_derivative);
            let cross_product_magnitude = cross_product.magnitude();
            let first_derivative_magnitude = first_derivative.magnitude();

            if first_derivative_magnitude == 0.0 {
                0.0
            } else {
                cross_product_magnitude / first_derivative_magnitude.powi(3)
            }
        }
        pub fn radius_of_curvature(&self, t: f64) -> f64 {
            let curvature = self.curvature(t);
            if curvature == 0.0 {
                f64::INFINITY // Infinite radius for zero curvature (straight line)
            } else {
                1.0 / curvature
            }
        }
        /// Evaluate a NURBS curve at parameter t using the De Boor algorithm
        pub fn evaluate(&self, t: f64) -> Point3d {
            let p = self.degree;

            // Find the knot span
            let k = self.find_span(t);

            // Allocate the working points array
            let mut d = vec![
                Point3d {
                    X: 0.0,
                    Y: 0.0,
                    Z: 0.0,
                };
                p + 1
            ];

            // Initialize the points for the De Boor recursion
            for j in 0..=p {
                let cp = &self.control_points[k - p + j];
                let w = self.weights[k - p + j];
                d[j] = Point3d {
                    X: cp.X * w,
                    Y: cp.Y * w,
                    Z: cp.Z * w,
                };
            }

            // Perform the De Boor recursion
            for r in 1..=p {
                for j in (r..=p).rev() {
                    let alpha = (t - self.knots[k + j - p])
                        / (self.knots[k + 1 + j - r] - self.knots[k + j - p]);

                    d[j].X = (1.0 - alpha) * d[j - 1].X + alpha * d[j].X;
                    d[j].Y = (1.0 - alpha) * d[j - 1].Y + alpha * d[j].Y;
                    d[j].Z = (1.0 - alpha) * d[j - 1].Z + alpha * d[j].Z;
                }
            }

            // Normalize by the weight
            let weight = self.weights[k];
            Point3d {
                X: d[p].X / weight,
                Y: d[p].Y / weight,
                Z: d[p].Z / weight,
            }
        }

        pub fn find_span(&self, t: f64) -> usize {
            let n = self.control_points.len() - 1;
            if t >= self.knots[n + 1] {
                return n;
            }
            if t <= self.knots[self.degree] {
                return self.degree;
            }
            for i in self.degree..=n {
                if t >= self.knots[i] && t < self.knots[i + 1] {
                    return i;
                }
            }
            n // Default case
        }

        /// Describe the path by a list of Point3d.
        pub fn draw(&self, step_resolution: f64) -> Vec<Point3d> {
            if step_resolution == 0.0 {
                panic!("Step resolution cannot be 0.0.")
            }
            let mut curve_points = Vec::new();
            let mut t = 0.0;
            while t <= 1.0 {
                curve_points.push(self.evaluate(t));
                t += step_resolution;
            }
            curve_points
        }
        pub fn numerical_first_derivative(&self, t: f64, h: f64) -> Point3d {
            let pt_plus_h = self.evaluate(t + h);
            let pt_minus_h = self.evaluate(t - h);
            Point3d::new(
                (pt_plus_h.X - pt_minus_h.X) / (2.0 * h),
                (pt_plus_h.Y - pt_minus_h.Y) / (2.0 * h),
                (pt_plus_h.Z - pt_minus_h.Z) / (2.0 * h),
            )
        }
        pub fn numerical_second_derivative(&self, t: f64, h: f64) -> Point3d {
            let pt_plus_h = self.evaluate(t + h);
            let pt = self.evaluate(t);
            let pt_minus_h = self.evaluate(t - h);
            Point3d::new(
                (pt_plus_h.X - 2.0 * pt.X + pt_minus_h.X) / (h * h),
                (pt_plus_h.Y - 2.0 * pt.Y + pt_minus_h.Y) / (h * h),
                (pt_plus_h.Z - 2.0 * pt.Z + pt_minus_h.Z) / (h * h),
            )
        }

        pub fn new_rh(control_points: Vec<Point3d>, degree: usize) -> Self {
            let n = control_points.len(); // Number of control points
            let num_knots = n + degree + 1; // Number of knots = control points + degree + 1

            // Generate a clamped knot vector
            let mut knots = vec![0.0; num_knots];

            // First and last degree + 1 knots are clamped to 0.0 and 1.0
            for i in 0..=degree {
                knots[i] = 0.0; // Clamped at the start
                knots[num_knots - 1 - i] = 1.0; // Clamped at the end
            }

            // Interior knots - number of interior knots is n - degree
            let interior_knots = n - degree;

            // Compute the interior knots
            for i in 1..interior_knots {
                let value = i as f64 / interior_knots as f64;
                knots[degree + i] = value;
            }

            // Default weights (all 1.0)
            let weights = vec![1.0; n];

            // Construct the NurbsCurve
            NurbsCurve {
                control_points,
                degree,
                knots,
                weights,
            }
        }
    }
    // Other existing methods...
    use std::fmt;
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Vector2d {
        pub x: f64,
        pub y: f64,
    }
    impl Vector2d {
        pub fn new(x: f64, y: f64) -> Self {
            Self { x, y }
        }
        pub fn magnetude(self) -> f64 {
            (self.x * self.x + self.y * self.y).sqrt()
        }
        pub fn normalize(self) -> Self {
            let m = self.magnetude();
            if m > std::f64::EPSILON {
                Self {
                    x: self.x / m,
                    y: self.y / m,
                }
            } else {
                Self { x: 0.0, y: 0.0 }
            }
        }
        pub fn crossProduct(first_vector: &Vector2d, second_vector: &Vector2d) -> f64 {
            ((*first_vector).x * (*second_vector).y) - ((*first_vector).y * (*second_vector).x)
        }

        pub fn angle(first_vector: &Vector2d, second_vector: &Vector2d) -> f64 {
            f64::acos(
                ((*first_vector) * (*second_vector))
                    / ((*first_vector).magnetude() * (*second_vector).magnetude()),
            )
        }
    }
    impl fmt::Display for Vector2d {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Vector2d{{x:{0},y:{1}}}", self.x, self.y)
        }
    }
    impl Sub for Vector2d {
        type Output = Self;
        fn sub(self, other: Vector2d) -> Self::Output {
            Self {
                x: self.x - other.x,
                y: self.y - other.y,
            }
        }
    }

    impl Add for Vector2d {
        type Output = Self;
        fn add(self, other: Vector2d) -> Self::Output {
            Self {
                x: self.x + other.x,
                y: self.y + other.y,
            }
        }
    }
    impl Mul<f64> for Vector2d {
        type Output = Self;
        fn mul(self, scalar: f64) -> Self::Output {
            Self {
                x: self.x * scalar,
                y: self.y * scalar,
            }
        }
    }
    impl Mul for Vector2d {
        type Output = f64;
        fn mul(self, other: Vector2d) -> f64 {
            self.x * other.x + self.y * other.y
        }
    }
    impl Div for Vector2d {
        type Output = Self;
        fn div(self, other: Vector2d) -> Self::Output {
            Self {
                x: self.x / other.x,
                y: self.y / other.y,
            }
        }
    }
    impl MulAssign<f64> for Vector2d {
        fn mul_assign(&mut self, scalar: f64) {
            self.x *= scalar;
            self.y *= scalar;
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub struct Quaternion {
        pub w: f64,
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    impl Quaternion {
        pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
            Self { w, x, y, z }
        }

        // Normalize the quaternion
        pub fn normalize(&self) -> Self {
            let mag =
                (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
            Self {
                w: self.w / mag,
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            }
        }

        // Quaternion multiplication
        pub fn multiply(&self, other: &Self) -> Self {
            Self {
                w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            }
        }

        // Quaternion conjugate
        pub fn conjugate(&self) -> Self {
            Self {
                w: self.w,
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        }

        // Rotate a point using the quaternion
        pub fn rotate_point(&self, point: &Point3d) -> Point3d {
            let q_point = Quaternion::new(0.0, point.X, point.Y, point.Z);
            let q_conjugate = self.conjugate();
            let rotated_q = self.multiply(&q_point).multiply(&q_conjugate);

            Point3d::new(rotated_q.x, rotated_q.y, rotated_q.z)
        }

        pub fn rotate_point_around_axis(point: &Point3d, axis: &Point3d, angle_rad: f64) -> Point3d {
            // Normalize the axis vector
            let axis_length = (axis.X * axis.X + axis.Y * axis.Y + axis.Z * axis.Z).sqrt();
            let axis_normalized = Point3d::new(
                axis.X / axis_length,
                axis.Y / axis_length,
                axis.Z / axis_length,
            );

            // Create the quaternion for the rotation
            let half_angle = angle_rad / 2.0;
            let sin_half_angle = half_angle.sin();
            let rotation_quat = Quaternion::new(
                half_angle.cos(),
                axis_normalized.X * sin_half_angle,
                axis_normalized.Y * sin_half_angle,
                axis_normalized.Z * sin_half_angle,
            )
            .normalize();

            // Rotate the point using the quaternion
            rotation_quat.rotate_point(point)
        }

        /// Convert the quaternion to a 4x4 transformation matrix
        pub fn to_4x4_matrix(&self) -> [[f64; 4]; 4] {
            let (w, x, y, z) = (self.w, self.x, self.y, self.z);

            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                    0.0,
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                    0.0,
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                    0.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }

        /// Convert the quaternion to a 4x3 transformation matrix
        pub fn to_4x3_matrix(&self) -> [[f64; 3]; 4] {
            let (w, x, y, z) = (self.w, self.x, self.y, self.z);

            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
                [0.0, 0.0, 0.0], // This row represents translation or is left blank for rotation-only matrices.
            ]
        }
    }
}

pub mod intersection {
    use super::geometry::{CPlane, Point3d, Vector3d};
    /// Compute intersection of two point projected by two vectors
    /// # Arguments
    /// p1 first points (Point3d), d1 first direction (Vector3d)
    /// p2 first points (Point3d), d2 first direction (Vector3d)
    /// # Returns
    /// None if vectors never intersect or Point3d of intersection on Success.
    /// ! vectors must be coplanar for intersecting.
    pub fn compute_intersection_coplanar(
        p1: &Point3d,
        d1: &Vector3d,
        p2: &Point3d,
        d2: &Vector3d,
    ) -> Option<Point3d> {
        let cross_d1_d2 = Vector3d::cross_product(d1, d2);
        let denom = cross_d1_d2 * cross_d1_d2; // dot product (square of cross product vector)
        if (f64::abs(denom) == 0f64) && !Vector3d::are_coplanar_a(d1, d2) {
            None // if lines never intersect.
        } else {
            let diff = *p2 - *p1; // Make vector delta.
            let t1 = Vector3d::cross_product(&diff, d2) * cross_d1_d2 / denom; // Compute intersection from formula.
            Some(*p1 + ((*d1) * t1)) // Return result.
        }
    }

    /// Project a Point3d on a CPlane.
    /// # Arguments
    /// a Point3d , a Vector3d and a CPlane.
    /// # Returns
    /// return a Point3d on success None if vector don't point on the plane.
    pub fn intersect_ray_with_plane(
        point: &Point3d,      // Starting point of the line (P0)
        direction: &Vector3d, // Direction vector of the line (V)
        plane: &CPlane,
    ) -> Option<Point3d> {
        let p0_to_plane = Vector3d::new(
            point.X - plane.origin.X,
            point.Y - plane.origin.Y,
            point.Z - plane.origin.Z,
        );

        let numerator = -(plane.normal * p0_to_plane);
        let denominator = plane.normal * (*direction);

        if denominator.abs() < 1e-6 {
            // The line is parallel to the plane (no intersection or line lies on the plane)
            return None;
        }

        // Compute t
        let t = numerator / denominator;

        // Compute intersection point
        let intersection = (*point) + ((*direction) * t);
        Some(intersection)
    }
}

/*
 * 'atomic' Case study for a simple representation of a 3d point
 *  on simple 2D screen via projection matrix honestly for now it's kind of
 *  a magic black box for me but let's refine the analysis.
 */
pub mod visualization_disabled_alpha {

    use super::geometry::{Point3d, Vector3d};

    pub struct Camera {
        position: Point3d, // Camera position in world space
        target: Point3d,   // The point the camera is looking at
        up: Vector3d,      // The "up" direction (usually the Y-axis)
        fov: f64,          // Field of view (in degrees)
        width: f64,        // Screen width
        height: f64,       // Screen height
        near: f64,         // Near clipping plane
        far: f64,          // Far clipping plane
    }

    impl Camera {
        pub fn new(
            position: Point3d,
            target: Point3d,
            up: Vector3d,
            width: f64,
            height: f64,
            fov: f64,
            near: f64,
            far: f64,
        ) -> Self {
            Self {
                position,
                target,
                up,
                fov,
                width,
                height,
                near,
                far,
            }
        }

        /// Get view matrix.
        fn get_view_matrix(&self) -> [[f64; 4]; 4] {
            let forward = Vector3d::new(
                self.position.X - self.target.X,
                self.position.Y - self.target.Y,
                self.position.Z - self.target.Z,
            )
            .unitize_b();
            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();
            let up = Vector3d::cross_product(&right, &forward).unitize_b();
            // a Point3d is used there instead of Vector3d to avoid
            // computing unused vector length automatically.
            let translation = Point3d::new(-self.position.X, -self.position.Y, -self.position.Z);
            [
                [right.get_X(), up.get_X(), -forward.get_X(), 0.0],
                [right.get_Y(), up.get_Y(), -forward.get_Y(), 0.0],
                [right.get_Z(), up.get_Z(), -forward.get_Z(), 0.0],
                // Compute dot product manually... (i don't plan to implemented 'Vector3d*Point3d').
                [
                    (right.get_X() * translation.X)
                        + (right.get_Y() * translation.Y)
                        + (right.get_Z() * translation.Z),
                    (up.get_X() * translation.X)
                        + (up.get_Y() * translation.Y)
                        + (right.get_Z() * translation.Z),
                    (forward.get_X() * translation.X)
                        + (forward.get_Y() * translation.Y)
                        + (forward.get_Z() * translation.Z),
                    1.0,
                ],
            ]
        }

        // Create a perspective projection matrix
        fn get_projection_matrix(&self) -> [[f64; 4]; 4] {
            let aspect_ratio = self.width / self.height;
            let fov_rad = self.fov.to_radians();
            let f = 1.0 / (fov_rad / 2.0).tan();
            [
                [f / aspect_ratio, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    (self.far + self.near) / (self.near - self.far),
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    (2.0 * self.far * self.near) / (self.near - self.far),
                    0.0,
                ],
            ]
        }

        // Project a 3D point to 2D space using the camera's view and projection matrices
        pub fn project(&self, point: Point3d) -> Option<(usize, usize)> {
            let view_matrix = self.get_view_matrix();
            let projection_matrix = self.get_projection_matrix();

            let camera_space_point = self.multiply_matrix_vector(view_matrix, point);
            let projected_point =
                self.multiply_matrix_vector(projection_matrix, camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.X / projected_point.Z;
            let y = projected_point.Y / projected_point.Z;

            // Map the coordinates from [-1, 1] to screen space
            let screen_x = ((x + 1.0) * 0.5 * self.width) as isize;
            let screen_y = ((1.0 - y) * 0.5 * self.height) as isize;

            if screen_x < 0
                || screen_x >= self.width as isize
                || screen_y < 0
                || screen_y >= self.height as isize
            {
                return None; // Point is out of screen bounds
            }

            Some((screen_x as usize, screen_y as usize))
        }

        pub fn project_with_depth(&self, point: Point3d) -> Option<(usize, usize, f64)> {
            let view_matrix = self.get_view_matrix();
            let projection_matrix = self.get_projection_matrix();

            // Step 1: Transform the point into camera space
            let camera_space_point = self.multiply_matrix_vector(view_matrix, point);

            // Extract depth in camera space (before projection)
            let depth_in_camera_space = camera_space_point.Z;

            // Step 2: Project the point using the projection matrix
            let projected_point =
                self.multiply_matrix_vector(projection_matrix, camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.X / projected_point.Z;
            let y = projected_point.Y / projected_point.Z;

            // Extract normalized depth from the projection
            //let _depth_in_ndc = projected_point.Z;

            // Map the coordinates from [-1, 1] to screen space
            let screen_x = ((x + 1.0) * 0.5 * self.width) as isize;
            let screen_y = ((1.0 - y) * 0.5 * self.height) as isize;

            if screen_x < 0
                || screen_x >= self.width as isize
                || screen_y < 0
                || screen_y >= self.height as isize
            {
                return None; // Point is out of screen bounds
            }

            // Return screen coordinates and the depth
            Some((screen_x as usize, screen_y as usize, depth_in_camera_space))
        }

        // Multiply 3d point by a matrix vector.
        fn multiply_matrix_vector(&self, matrix: [[f64; 4]; 4], v: Point3d) -> Point3d {
            Point3d::new(
                matrix[0][0] * v.X + matrix[0][1] * v.Y + matrix[0][2] * v.Z + matrix[0][3],
                matrix[1][0] * v.X + matrix[1][1] * v.Y + matrix[1][2] * v.Z + matrix[1][3],
                matrix[2][0] * v.X + matrix[2][1] * v.Y + matrix[2][2] * v.Z + matrix[2][3],
            )
        }
    }
}

/*
 *  A set of early very basic transformations method
 *  of Point3d from world axis and Angles.
 */
pub mod transformation {
    use super::geometry::Point3d;
    use crate::display_pipe_line::redering_object::Vertex;
    /// Rotate the point from Y world axis.
    /// # Arguments
    /// Point3d to transform and angle in radian (in f64)
    /// # Returns
    /// return a Point3d
    pub fn rotate_y(point: Point3d, angle: f64) -> Point3d {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Point3d {
            X: point.X * cos_theta - point.Z * sin_theta,
            Y: point.Y,
            Z: point.X * sin_theta + point.Z * cos_theta,
        }
    }
    pub fn rotate_y_vertex(point: Vertex, angle: f64) -> Vertex {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Vertex {
            x: point.x * cos_theta - point.z * sin_theta,
            y: point.y,
            z: point.x * sin_theta + point.z * cos_theta,
        }
    }

    pub fn rotate_x(point: Point3d, angle: f64) -> Point3d {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Point3d {
            X: point.X,
            Y: point.Y * cos_theta - point.Z * sin_theta,
            Z: point.Y * sin_theta + point.Z * cos_theta,
        }
    }
    pub fn rotate_x_vertex(point: Vertex, angle: f64) -> Vertex{
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Vertex {
            x: point.x,
            y: point.y * cos_theta - point.z * sin_theta,
            z: point.y * sin_theta + point.z * cos_theta,
        }
    }
    pub fn rotate_z(point: Point3d, angle: f64) -> Point3d {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Point3d {
            X: point.X * cos_theta - point.Y * sin_theta,
            Y: point.X * sin_theta + point.Y * cos_theta,
            Z: point.Z,
        }
    }
    pub fn rotate_z_vertex(point: Vertex, angle: f64) -> Vertex {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Vertex {
            x: point.x * cos_theta - point.y * sin_theta,
            y: point.x * sin_theta + point.y * cos_theta,
            z: point.z,
        }
    }

    /// Project a 3d point on a 4 points3d plane (from the plane Vector Normal)
    pub fn project_3d_point_on_plane(point: &Point3d, plane_pt: &[Point3d; 4]) -> Option<Point3d> {
        // Make a plane vectors from inputs points.
        let plane = [
            (*plane_pt)[0] - (*plane_pt)[1],
            (*plane_pt)[3] - (*plane_pt)[0],
        ];
        if let Some(projection) = ((*point) - (*plane_pt)[0]).project_on_infinite_plane(&plane) {
            let result_point = (*plane_pt)[0] + projection;
            // Test if point is on plane.
            if result_point.is_on_plane(&plane_pt) {
                Some(result_point)
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub mod draw {
    // Bresenham's line algorithm.
    // Draw a line between two 2d points on screen.
    // it's a clever algorithm dynamically plotting
    // the distance between two points.
    // - Bresenham's algorithm compute at each loop the direction (x,y)
    //   of the next 2d point to plot and so, draw a line in
    //   a (x,y) space and efficiently create the illusion of
    //   a 3d line moving or rotating
    //   (if created with a 3d point projected in 2d).
    pub fn draw_line(
        buffer: &mut Vec<u32>,
        width: usize,
        start: (usize, usize),
        end: (usize, usize),
        color: u32,
    ) {
        // Assign reference for readability
        let (x0, y0) = (start.0 as isize, start.1 as isize);
        let (x1, y1) = (end.0 as isize, end.1 as isize);

        // Compute absolute distance difference.
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();

        // Evaluate the (x and y) direction.
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };

        // Make mutable copy for iteration.
        let mut err = dx - dy;
        let mut x = x0;
        let mut y = y0;

        loop {
            // Write the pixel color value in the mutable buffer (memory block) at x and y position offset
            // this if screen matrice condition are met (see memory block allocation syntax for cursor positioning)
            if x >= 0 && x < width as isize && y >= 0 && y < (buffer.len() / width) as isize {
                buffer[y as usize * width + x as usize] = color;
            }
            // Stop at end (when start point reach end point (x,y).
            if x == x1 && y == y1 {
                break;
            }
            // Evaluate the next x and y direction (we are only in 2D space).
            // algorithm key point: is to double the reminder by two
            // allowing to shrink the reminder by dy and dx in one step
            // making Bresenham's line algorithm efficient.
            // at shrinking the line distance to compute towards the endpoint
            // on x and y right direction for a given orientation.
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy; // shrink reminder
                x += sx; // adjust x cursor position (+/-)1
            }
            if e2 < dx {
                err += dx; // shrink reminder
                y += sy; // adjust y cursor position (+/-)1
            }
        }
    }

    use core::f64;

    use super::geometry::Point3d;
    use crate::models_3d::FONT_5X7;

    pub fn draw_text(
        buffer: &mut Vec<u32>,
        height: &usize,
        width: &usize,
        x: &usize,
        y: &usize,
        text: &str,
        scale: &usize,
        text_color: &u32,
    ) {
        for (i, c) in text.chars().enumerate() {
            let index = match c {
                'A'..='Z' => c as usize - 'A' as usize,
                'a'..='z' => c as usize - 'a' as usize + 26,
                '0'..='9' => c as usize - '0' as usize + 52,
                ' ' => 62, // Space character index
                '!' => 63,
                '"' => 64,
                '#' => 65,
                '$' => 66,
                '%' => 67,
                '&' => 68,
                '\'' => 69,
                '(' => 70,
                ')' => 71,
                '*' => 72,
                '+' => 73,
                ',' => 74,
                '-' => 75,
                '.' => 76,
                '/' => 77,
                ':' => 78,
                _ => continue, // Ignore unsupported characters
            };
            let char_data = &FONT_5X7[index];
            for (row, &row_data) in char_data.iter().enumerate() {
                for col in 0..5 {
                    if (row_data & (1 << (4 - col))) != 0 {
                        let px = x + i * 6 * scale + col * scale; // Adjust for character spacing and scaling
                        let py = y + row * scale;

                        // Draw scaled pixel
                        for dy in 0..(*scale) {
                            for dx in 0..(*scale) {
                                let sx = px + dx;
                                let sy = py + dy;
                                if sx < *width && sy < *height {
                                    buffer[sy * width + sx] = *text_color; // White pixel
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /*
     * Draw a contextual Grid representing the gradual depth in
     * a static place in world Coordinates
     * ( providing relative orientation to world when every
     *  thing will spin or move while in conception phase
     *  or visual evaluation phase ).
     */
    use super::geometry::CPlane;
    pub fn draw_3d_grid(
        plane: &CPlane,
        x_max: &f64,
        y_max: &f64,
        grid_spacing_unit: &f64,
    ) -> Vec<Point3d> {
        let mut grid_points = Vec::new();
        let grid_unit = grid_spacing_unit / x_max;
        let mut x = 0.0;
        let mut y = 0.0;
        while x <= *x_max {
            while y <= *y_max {
                grid_points.push((*plane).point_on_plane_uv(x, y));
                y += grid_unit;
            }
            if y >= *y_max {
                y = 0.0;
            }
            x += grid_unit;
        }
        grid_points
    }

    /// Draw a circle.
    pub fn draw_3d_circle(origin: Point3d, radius: f64, step: f64) -> Vec<Point3d> {
        let mut increm = 0.0f64;
        let mut circle_pts = Vec::new();
        while increm <= (f64::consts::PI * 2.0) {
            circle_pts.push(Point3d::new(
                (f64::sin(increm) * radius) + origin.X,
                (f64::cos(increm) * radius) + origin.Y,
                0.0 + origin.Z,
            ));
            increm += (f64::consts::PI * 2.0) / step;
        }
        circle_pts
    }

    // A 2D point with (x, y)
    #[derive(Debug, Clone, Copy)]
    pub struct Point2D {
        pub x: f64,
        pub y: f64,
    }

    pub fn draw_triangle_2d_v2(
        buffer: &mut Vec<u32>,
        width: usize,
        height: usize,
        p0: &Point2D,
        p1: &Point2D,
        p2: &Point2D,
        color: u32,
    ) {
        // Sort vertices by y-coordinate
        let mut points = [p0, p1, p2];
        points.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
        let (p0, p1, p2) = (points[0], points[1], points[2]);

        // Initialize a scanline table to store x-coordinates for each y
        let mut scanline_x: Vec<Vec<usize>> = vec![Vec::new(); height];

        // Rasterize triangle edges
        rasterize_edge(p0, p1, &mut scanline_x);
        rasterize_edge(p1, p2, &mut scanline_x);
        rasterize_edge(p0, p2, &mut scanline_x);

        // Fill the triangle by drawing horizontal spans
        for (y, x_coords) in scanline_x.iter().enumerate() {
            if x_coords.is_empty() || y >= height {
                continue;
            }

            // Sort the x-coordinates and fill between the leftmost and rightmost points
            let mut x_coords = x_coords.clone();
            x_coords.sort_unstable();
            let x_start = x_coords[0];
            let x_end = x_coords[x_coords.len() - 1];

            for x in x_start..=x_end {
                if x < width {
                    buffer[y * width + x] = color;
                }
            }
        }
    }
    pub fn rasterize_edge(
        edge_start: &Point2D,
        edge_end: &Point2D,
        scanline_x: &mut Vec<Vec<usize>>,
    ) {
        let mut x0 = edge_start.x as isize;
        let mut y0 = edge_start.y as isize;
        let x1 = edge_end.x as isize;
        let y1 = edge_end.y as isize;

        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        while x0 != x1 || y0 != y1 {
            // Store x for the current y
            if (y0 as usize) < scanline_x.len() {
                scanline_x[y0 as usize].push(x0 as usize);
            }

            let e2 = err * 2;
            if e2 > -dy {
                err -= dy;
                x0 += sx;
            }
            if e2 < dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    #[derive(Copy, Clone)]
    pub struct Edge {
        pub y_min: usize,
        pub y_max: usize,
        pub x: f64,
        pub dx: f64, // Increment in x per scanline
    }

    fn build_edges(p0: &Point2D, p1: &Point2D) -> Option<Edge> {
        let (top, bottom) = if p0.y < p1.y { (p0, p1) } else { (p1, p0) };

        let dy = bottom.y - top.y;
        if dy == 0.0 {
            return None; // Horizontal line, skip
        }

        let dx = (bottom.x - top.x) / dy; // Slope (dx/dy)

        Some(Edge {
            y_min: top.y as usize,
            y_max: bottom.y as usize,
            x: top.x,
            dx,
        })
    }
    pub fn draw_triangle_optimized(
        buffer: &mut Vec<u32>,
        width: usize,
        height: usize,
        p0: &Point2D,
        p1: &Point2D,
        p2: &Point2D,
        color: u32,
    ) {
        // Sort vertices by y-coordinate
        let mut points = [p0, p1, p2];
        points.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());
        let (p0, p1, p2) = (points[0], points[1], points[2]);

        // Build edges
        let mut edges = vec![];
        if let Some(edge) = build_edges(p0, p1) {
            edges.push(edge);
        }
        if let Some(edge) = build_edges(p1, p2) {
            edges.push(edge);
        }
        if let Some(edge) = build_edges(p0, p2) {
            edges.push(edge);
        }

        // Sort edges by y_min
        edges.sort_by(|a, b| a.y_min.cmp(&b.y_min));

        // Active edge list (AEL)
        let mut active_edges: Vec<Edge> = vec![];

        // Scanline processing
        for y in 0..height {
            // Add edges to the active list
            for i in 0..edges.len() {
                if edges[i].y_min == y {
                    active_edges.push(edges[i]);
                }
            }

            // Remove edges that go out of scope
            active_edges.retain(|e| e.y_max > y);

            // Sort active edges by `x` coordinate
            active_edges.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

            // Fill spans between edge pairs
            for pair in active_edges.chunks(2) {
                if let [left, right] = pair {
                    let x_start = left.x as usize;
                    let x_end = right.x as usize;

                    for x in x_start..=x_end {
                        if x < width {
                            buffer[y * width + x] = color;
                        }
                    }
                }
            }

            // Update x-coordinates for active edges
            for edge in active_edges.iter_mut() {
                edge.x += edge.dx;
            }
        }
    }
}

pub mod utillity {
    use core::f64;
    //Rust have already builtin function.
    pub fn degree_to_radians(input_angle_in_degre: &f64) -> f64 {
        (*input_angle_in_degre) * (f64::consts::PI * 2.0) / 360.0
    }
    pub fn radians_to_degree(input_angle_in_radians: &f64) -> f64 {
        (*input_angle_in_radians) * 360.0 / (f64::consts::PI * 2.0)
    }

    ///The famous Quake3 Arena algorithm.
    pub fn fast_inverse_square_root(x: f32) -> f32 {
        let threehalfs: f32 = 1.5;

        let x2: f32 = x * 0.5;
        let mut y: f32 = x;

        let mut i: i32 = y.to_bits() as i32; // Get bits for floating value
        i = 0x5f3759df - (i >> 1); // What the algorithm does
        y = f32::from_bits(i as u32); // Convert bits back to float

        y = y * (threehalfs - (x2 * y * y)); // 1st iteration (can repeat for accuracy)

        y
    }
    /// Remap range 1 to range 2 from s value at the scale of range 1.
    pub fn remap(from_range: (f64, f64), to_range: (f64, f64), s: f64) -> f64 {
        to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
    }

    use std::ops::{Div, Sub};
    /// inverse linear interpolation... Normalize a range...
    /// used to find the relative position (between 0 and 1)
    /// of a value within a given range. This is useful for normalizing values.
    /// # Returns
    /// an Option<T> a normalized (t) parametes value from 0 to 1
    /// describing the interval from v_start to v_end.
    /// input is not clamped so the range will exceed interval linearly.
    /// T gneric can be f64 usize i64 or what ever implementing Sub and Div and Copy traits.
    pub fn ilerp<T: std::cmp::PartialEq>(v_start: T, v_end: T, value: T) -> Option<T>
    where
        T: Copy + Sub<Output = T> + Div<Output = T>,
    {
        if v_start == v_end {
            None
        } else {
            Some((value - v_start) / (v_end - v_start))
        }
    }

    /// # Returns
    /// return the linears interpolation between two values
    /// from a t normalized f64 parameters from 0.0 to 1.0
    pub fn lerp<T>(a: T, b: T, t: f64) -> T
    where
        T: Copy
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<f64, Output = T>,
    {
        a + (b - a) * t
    }
}

#[cfg(test)]
mod test {
    use super::geometry::*;
    #[test]
    fn test_cross_product() {
        let vec_a = Vector3d::new(1.0, 0.0, 0.0);
        let vec_b = Vector3d::new(0.0, 1.0, 0.0);
        assert_eq!(
            Vector3d::new(0.0, 0.0, 1.0),
            Vector3d::cross_product(&vec_a, &vec_b)
        );
    }
    #[test]
    fn test_vector3d_negation() {
        let v_totest = Vector3d::new(0.0, -0.35, 8.0);
        let v_tocompare = Vector3d::new(0.0, 0.35, -8.0);
        assert_eq!(v_totest, -v_tocompare);
    }
    #[test]
    fn test_vector3d_length() {
        assert_eq!(f64::sqrt(2.0), Vector3d::new(1.0, 1.0, 0.0).Length());
    }
    #[test]
    fn test_vector3d_unitize() {
        let mut vector = Vector3d::new(6.0, 2.0, 8.0);
        vector.unitize();
        assert_eq!(1.0, vector.Length());
    }
    #[test]
    fn test_vector3d_scalar_a() {
        let mut vector = Vector3d::new(6.0, 2.0, 8.0);
        vector.unitize();
        vector = vector * 4.0;
        assert_eq!(4.0, vector.Length());
    }
    #[test]
    fn test_vector3d_scalar_b() {
        let mut vector = Vector3d::new(6.0, 2.0, 8.0);
        vector.unitize();
        vector *= 4.0;
        assert_eq!(4.0, vector.Length());
    }

    #[test]
    fn test_vector_is_same_direction() {
        let v1 = Vector3d::new(0.426427, -0.904522, 0.0);
        let v2 = Vector3d::new(0.688525, -0.7255212, 0.0);
        assert!(v1.is_same_direction(&v2, 0.94));
    }
    use core::f64;
    use std::f64::consts::PI;
    #[test]
    fn test_vector3d_angle() {
        let v1 = Vector3d::new(0.0, 1.0, 0.0);
        let v2 = Vector3d::new(1.0, 0.0, 0.0);
        assert_eq!(PI / 2.0, Vector3d::compute_angle(&v1, &v2));
    }
    use super::intersection::*;
    #[test]
    fn test_intersection_a() {
        let p1 = Point3d::new(0.0, 1.0, 0.0);
        let d1 = Vector3d::new(0.0, -1.0, 0.0);
        let p2 = Point3d::new(1.0, 0.0, 0.0);
        let d2 = Vector3d::new(-1.0, 0.0, 0.0);
        assert_eq!(
            Point3d::new(0.0, 0.0, 0.0),
            compute_intersection_coplanar(&p1, &d1, &p2, &d2).unwrap()
        );
    }
    #[test]
    fn test_intersection_b() {
        let p1 = Point3d::new(82.157, 30.323, 0.0);
        let d1 = Vector3d::new(0.643, -0.766, 0.0);
        let p2 = Point3d::new(80.09, 19.487, 0.0);
        let d2 = Vector3d::new(0.94, 0.342, 0.0);
        let expected_result = Point3d::new(88.641361, 22.59824, 0.0);
        if let Some(result) = compute_intersection_coplanar(&p1, &d1, &p2, &d2) {
            if (result - expected_result).Length() < 26e-6 {
                assert!(true);
            } else {
                assert!(false)
            }
        }
    }
    #[test]
    fn test_coplanar_vectors_a() {
        let pt1 = Point3d::new(82.832047, 36.102125, -3.214695);
        let pt2 = Point3d::new(85.341596, 34.653236, -2.539067);
        let pt3 = Point3d::new(82.0, 34.0, -4.040822);
        let pt4 = Point3d::new(85.0, 34.0, -2.82932);
        let v1 = pt2 - pt1;
        let v2 = pt4 - pt3;
        let v3 = pt3 - pt1;
        assert_eq!(true, Vector3d::are_coplanar_b(&v1, &v2, &v3));
    }

    #[test]
    fn test_coplanar_vectors_b() {
        let pt1 = Point3d::new(82.832047, 36.102125, -3.214695);
        let pt2 = Point3d::new(85.341596, 34.653236, -2.139067); // (this point is changed)
        let pt3 = Point3d::new(82.0, 34.0, -4.040822);
        let pt4 = Point3d::new(85.0, 34.0, -2.82932);
        let v1 = pt2 - pt1;
        let v2 = pt4 - pt3;
        let v3 = pt3 - pt1;
        assert_eq!(false, Vector3d::are_coplanar_b(&v1, &v2, &v3));
    }

    #[test]
    fn test_rotated_vector_a() {
        let vector_a = Vector3d::new(0.0, 1.0, 0.0);
        let axis = Vector3d::new(0.0, 0.0, 1.0);
        let rotated_vector = vector_a.rotate_around_axis(&axis, PI / 4.0);
        // make a unit 45 deg vector.
        let expected_vector = Vector3d::new(1.0, 1.0, 0.0).unitize_b();
        if (expected_vector - rotated_vector).Length().abs() < 1e-6 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_rotated_vector_b() {
        let vector_a = Vector3d::new(-5.0, 4.0, 3.0);
        let axis = Vector3d::new(30.0, 3.0, 46.0);
        let rotated_vector = vector_a.rotate_around_axis(&axis, 1.047198);
        // make a unit 45 deg vector.
        let expected_vector = Vector3d::new(0.255535, 7.038693, -0.625699);
        //assert_eq!(rotated_vector,expected_vector);
        if (expected_vector - rotated_vector).Length().abs() < 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    
    #[test]
    fn test_vector3d_are_perpendicular(){
        let vec_a = Vector3d::new(1.3,1.55,2.4);
        let vec_b = Vector3d::new(0.9,1.25,1.11);
        let vec_c =  Vector3d::cross_product(&vec_a, &vec_b).unitize_b();
        assert!(Vector3d::are_perpandicular(&vec_a, &vec_c));
    }

    use super::transformation::project_3d_point_on_plane;
    #[test]
    fn test_project_point_on_plane() {
        let plane = [
            Point3d::new(0.0, 4.0, 7.0),
            Point3d::new(6.775301, 11.256076, 5.798063),
            Point3d::new(-2.169672, 21.088881, 9.471482),
            Point3d::new(-5.0, 9.0, 9.0),
        ];
        let pt_to_project = Point3d::new(-4.781863, 14.083874, 1.193872);
        let expected_result = Point3d::new(-2.571911, 13.271809, 8.748913);
        //assert_eq!(expected_result,project_3d_point_on_infinite_plane(&pt_to_project, &plane).unwrap());
        if let Some(point) = project_3d_point_on_plane(&pt_to_project, &plane) {
            if (point - expected_result).Length() < 1e-6 {
                assert!(true);
            } else {
                assert!(false);
            }
        }
    }

    #[test]
    fn test_if_point_is_on_plane() {
        let plane = [
            Point3d::new(0.0, 4.0, 7.0),
            Point3d::new(6.775301, 11.256076, 5.798063),
            Point3d::new(-2.169672, 21.088881, 9.471482),
            Point3d::new(-5.0, 9.0, 9.0),
        ];
        let point_to_test = Point3d::new(-2.571911, 13.271809, 8.748913);
        assert_eq!(true, point_to_test.is_on_plane(&plane));
    }

    #[test]
    fn test_local_point() {
        let plane_origin_pt = Point3d::new(4.330127, 10.0, -15.5);
        let plane_normal = Vector3d::new(0.5, 0.0, -0.866025);
        let point = Point3d::new(5.0, 5.0, 5.0);
        let expected_result = Point3d::new(2.5, 5.0, -22.330127);
        let cp = CPlane::new(&plane_origin_pt, &plane_normal);
        let result = cp.point_on_plane(point.X, point.Y, point.Z);
        // assert_eq!(expected_result,result);
        if (expected_result - result).Length().abs() <= 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_degrees_to_radians() {
        use super::utillity::*;
        let angle_to_test = 90.0;
        assert_eq!(f64::consts::PI / 2.0, degree_to_radians(&angle_to_test));
    }

    #[test]
    fn test_radians_to_degrees() {
        use super::utillity::*;
        let angle_to_test = f64::consts::PI / 2.0;
        assert_eq!(90.0, radians_to_degree(&angle_to_test));
    }

    #[test]
    fn test_ray_trace_v_a() {
        use super::intersection::*;
        let point = Point3d::new(7.812578, 4.543698, 23.058283);
        let direction = Vector3d::new(-1.398849, 0.106953, -0.982613);
        let plane_origin = Point3d::new(-8.015905, -1.866453, 5.80651);
        let plane_normal = Vector3d::new(0.65694, -0.31293, 0.685934);
        let plane = CPlane::new(&plane_origin, &plane_normal);
        let expected_result = Point3d::new(-9.583205, 5.873738, 10.838721);
        // assert_eq!(expected_result,intersect_line_with_plane(&point, &direction, &plane).unwrap());
        if let Some(result_point) = intersect_ray_with_plane(&point, &direction, &plane) {
            if (result_point - expected_result).Length().abs() < 1e-5 {
                assert!(true);
            } else {
                assert!(false);
            }
        }
    }
    #[test]
    fn test_ray_trace_v_b() {
        //use super::visualization::redering_object::*;
        use crate::display_pipe_line::redering_object::*;

        // Define some triangles.
        let triangles = vec![
            Triangle::new(
                Vertex::new(0.0, 0.0, 0.0),
                Vertex::new(1.0, 0.0, 0.0),
                Vertex::new(0.0, 1.0, 0.0),
            ),
            Triangle::new(
                Vertex::new(1.0, 1.0, 1.0),
                Vertex::new(2.0, 1.0, 1.0),
                Vertex::new(1.0, 2.0, 1.0),
            ),
        ];

        // Spatial Acceleration with BVH tree.
        // Consist to build volumes box(s) around mesh triangles
        // to first evaluate rays intersections only with
        // that bounding box before weather or not digging
        // deeper in the mesh face itself with more rays.
        // box are build in tree structures where
        // - leafs are face Bounding box volumes,
        // - parent are the sum of the childrens bb volumes
        // - the root tree is the bounding box of the whole sub bb volumes.

        // Build the BVH (Bounding Volumes Hiearchy).
        let bvh = BVHNode::build(triangles, 0);

        // Define a ray.
        let origin = Vertex::new(0.5, 0.5, -1.0);
        let direction = Vertex::new(0.0, 0.0, 1.0);
        let ray = Ray::new(origin, direction);

        // Perform intersection test on bvh.
        if let Some((t, _triangle)) = bvh.intersect(&ray) {
            //  _triangle is the ref to intersected triangle geometry.
            // *_triangle.intersect(&ray) (for refinements...)
            println!("Hit triangle at t = {}!", t);
        } else {
            println!("No intersection.");
            assert!(false);
        }
        assert!(true);
    }

    use crate::display_pipe_line::visualization_v3::coloring::Color;
    #[test]
    fn test_color() {
        let red: u8 = 20;
        let green: u8 = 19;
        let blue: u8 = 20;
        assert_eq!(0x141314, Color::convert_rgb_color(red, green, blue));
    }
    #[test]
    fn test_import_export_obj_size_ligh() {
        let vertices = vec![
            Vertex::new(0.0, 0.0, 0.0),
            Vertex::new(1.0, 0.0, 1.0),
            Vertex::new(1.0, 1.0, 0.0),
            Vertex::new(0.0, 1.0, 0.2),
        ];
        let triangles = vec![
            Triangle::new(vertices[0], vertices[1], vertices[2]),
            Triangle::new(vertices[0], vertices[2], vertices[3]),
        ];
        let mesh = Mesh::new(vertices, triangles);
        mesh.export_to_obj("./geometry/exported_light_with_rust.obj")
            .ok();
        let expected_data = Mesh::count_obj_elements("./geometry/exported_light_with_rust.obj")
            .ok()
            .unwrap();
        let imported_mesh =
            Mesh::import_obj_with_normals("./geometry/exported_light_with_rust.obj").unwrap();
        assert_eq!(
            (expected_data.0, expected_data.2),
            (imported_mesh.vertices.len(), imported_mesh.triangles.len())
        );
    }
    #[test]
    fn test_import_export_obj_size_medium() {
        let expected_data = Mesh::count_obj_elements("./geometry/medium_geometry.obj")
            .ok()
            .unwrap();
        let imported_mesh =
            Mesh::import_obj_with_normals("./geometry/medium_geometry.obj").unwrap();
        imported_mesh
            .export_to_obj_with_normals_fast("./geometry/medium_geometry_exported_from_rust.obj")
            .ok();
        assert_eq!(
            (expected_data.0, expected_data.2),
            (imported_mesh.vertices.len(), imported_mesh.triangles.len())
        );
    }
    #[test]
    fn test_import_obj_size_hight() {
        let expected_data = Mesh::count_obj_elements("./geometry/hight_geometry.obj")
            .ok()
            .unwrap();
        let imported_mesh = Mesh::import_obj_with_normals("./geometry/hight_geometry.obj").unwrap();
        imported_mesh
            .export_to_obj_with_normals_fast("./geometry/hight_geometry_exported_from_rust.obj")
            .ok();
        assert_eq!(
            (expected_data.0, expected_data.2),
            (imported_mesh.vertices.len(), imported_mesh.triangles.len())
        );
    }
    #[test]
    fn test_nurbs_curve_a() {
        let cv = vec![
            Point3d::new(0.0, 0.0, 0.0),
            Point3d::new(0.0, 10.0, 0.0),
            Point3d::new(10.0, 10.0, 0.0),
            Point3d::new(10.0, 0.0, 0.0),
            Point3d::new(20.0, 0.0, 0.0),
            Point3d::new(20.0, 10.0, 0.0),
        ];
        let crv = NurbsCurve::new(cv, 5);
        assert_eq!(Point3d::new(10.0, 5.0, 0.0), crv.evaluate(0.5));
        let pt = crv.evaluate(0.638);
        let expected_result = Point3d::new(13.445996, 3.535805, 0.0);
        if (pt - expected_result).Length() < 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    #[test]
    fn test_nurbs_curve_b() {
        let cv = vec![
            Point3d::new(-42.000, -13.000, 0.000),
            Point3d::new(-27.300, -26.172, 22.000),
            Point3d::new(-23.034, 15.562, 0.000),
            Point3d::new(7.561, -14.082, -21.000),
            Point3d::new(8.000, -19.000, 0.000),
        ];
        let crv = NurbsCurve::new(cv, 4);
        let t = 0.3;
        if (21.995 - crv.radius_of_curvature(t)).abs() < 1e-3 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    use crate::display_pipe_line::redering_object::{Triangle, Vertex,Mesh};
    #[test]
    fn test_triangle_area() {
        // The following Triangle is flat in XY plane.
        let v1 = Vertex::new(1.834429, 0.0, -0.001996);
        let v2 = Vertex::new(1.975597, 0.0, 0.893012);
        let v3 = Vertex::new(2.579798, 0.0, 0.150466);
        let tri = Triangle::new(v1, v2, v3);
        let expected_reuslt_area = 0.322794;
        let result = tri.get_triangle_area();
        if (expected_reuslt_area - result).abs() < 1e-6 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn volume_mesh_test() {
        let obj = Mesh::import_obj_with_normals("./geometry/mesh_volume.obj").unwrap();
        let volume = obj.compute_volume();
        let expected_volume = 214.585113;
        if (volume - expected_volume).abs() < 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn volume_mesh_water_tight_a() {
        let obj = Mesh::import_obj_with_normals("./geometry/mesh_volume.obj").unwrap();
        if obj.is_watertight() {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    #[test]
    fn volume_mesh_water_tight_b() {
        let obj = Mesh::import_obj_with_normals("./geometry/non_water_tight.obj").unwrap();
        if !obj.is_watertight() {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    #[test]
    fn volume_mesh_water_tight_c_parrallel() {
        let obj = Mesh::import_obj_with_normals("./geometry/non_water_tight.obj").unwrap();
        if !obj.is_watertight_par() {
            assert!(true);
        } else {
            assert!(false);
        }
        let obj2 = Mesh::import_obj_with_normals("./geometry/mesh_volume.obj").unwrap();
        if obj2.is_watertight_par() {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    #[test]
    fn test_inside_a_mesh() {
        let obj = Mesh::import_obj_with_normals("./geometry/torus.obj").unwrap();
        let v1_to_test = Vertex::new(0.0, 0.0, 0.0); // should be outside the torus.
        let v2_to_test = Vertex::new(4.0, 0.0, 0.0); // should be inside.
        assert_eq!(false, v1_to_test.is_inside_a_mesh(&obj));
        assert_eq!(true, v2_to_test.is_inside_a_mesh(&obj));
    }
}
/*
 * notes:
 *     deltaTime = now - start "time par frame" (this keep runtime granularity out of the equation as much as possible).
 *     for realtime animation:
 *     Vector3d velocity += (Vector3d accelleration * deltaTime) // ifluance velocity over time
 *     Point3d position += (Vector3d Velocity * deltaTime) // this is in space unit/frame
 *
 *     - Accelleration with negative value can simulate gravity. (it's incrementally added to
 *       velocity vector over time with a Z negative value which flip smoothlly the polarity of the
 *       velocity vector.)
 *  
 *
 * notes: Fundemental for understanding basic of matrix rotation.
 *        -------------------------------------------------------
 *        - it would be crazy to study further without 
 *          a very basic understanding of fundementals.
 *        - it's important to have a 'solid definition' of your understanding 
 *          in face of such amazing and fundemental complex system.
 *          so describing formula with verbose is the only way to constraint my 
 *          mind on the basis.
 *        -------------------------------------------------------
 *     - Trigger Action matrix Entity:
 *          x' = x cos(theta) - y sin(theta), y' = x sin(theta) + y cos(theta),
 *
 *  Description:
 *      x' = x component of a rotated vector2d(x,y) from basis vectors system.
 *      x cos(theta) = component x on basis system * (time) cos(theta) -> equal to 1 if cos(0.0deg) 
 *      y sin(theta) = component y on basis system * (time) sin(theta) -> equal to 0 if sin(0.0deg)
 *      x cos(theta) - y sin(theta) = (1 * 1) - (0 * 0) if theta = 0.0 deg  
 *      --------------------------------------------------------------------
 *      y' = y component of a rotated vector2d(x,y) from basis vectors system.
 *      x sin(theta) =  component x on basis system * (time) sin(theta) -> equal to 0 if sin(0.0deg) 
 *      y cos(theta) =  component y on basis system * (time) cos(theta) -> equal to 0 if cos(0.0deg)
 *      x sin(theta) + y cos(theta) = (1 * 0) - (0 * 1) if theta = 0.0 deg  
 * 
 *  this describe mathematically the rotation of an unit vector on a orthogonal basis sytem.
 *  - core mechanisum cos(theta) and sin(theta) will serve to divide unit basis axis length 
 *  by multiply the basix length of reference (x or y) by a number from 0.0 to 1.0 giving 
 *  a division of (x,y) component of the rotated vector.
 *
 *  This produce: the 4x4 matrix rotation. 
 *  let rotation_z = [
                [z_rad.cos(), -z_rad.sin(), 0.0, 0.0],
                [z_rad.sin(), z_rad.cos(), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            // Rotation matrix around the X-axis
            let rotation_x = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, x_rad.cos(), -x_rad.sin(), 0.0],
                [0.0, x_rad.sin(), x_rad.cos(), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Rotation matrix around the Y-axis
            let rotation_y = [
                [y_rad.cos(), 0.0, y_rad.sin(), 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-y_rad.sin(), 0.0, y_rad.cos(), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
 *
 */
