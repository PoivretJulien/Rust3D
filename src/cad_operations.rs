// *************************************************************************
// *** Main objects of the lib representing all the classic  operations. ***
// *************************************************************************

#[allow(dead_code)]
pub mod geometry {
    use crate::render_tools::rendering_object::Vertex;
    use core::f64;
    use std::f64::EPSILON;
    use std::fmt;
    use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

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

        /// Test if a point is on a plane define by 4 corners points.
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

        /// Determines if a series of points lies on a straight line.
        pub fn are_points_collinear(points: &[Point3d], tolerance: f64) -> bool {
            if points.len() < 3 {
                // Less than 3 points are always collinear
                return true;
            }
            // Calculate the reference vector from the first two points
            let reference_vector = points[1] - points[0];
            for i in 2..points.len() {
                // Get the vector from the first point to the current point
                let current_vector = points[i] - points[0];
                // Compute the cross product of the reference vector and the current vector
                let cross_product = Vector3d::cross_product(&reference_vector, &current_vector);
                // If the cross product is not zero, points are not collinear
                if !cross_product.is_zero(tolerance) {
                    return false;
                }
            }
            true
        }
        /// find the first collinear index of an array of points.
        pub fn find_first_collinear_points(
            array_input: &[Point3d],
            tolerance: f64,
        ) -> Option<(usize, usize)> {
            // find the 3 first segments where the points array describe a straight line.
            if array_input.len() > 3 {
                let mut result = (0, 0);
                for i in 0..array_input.len() - 4 {
                    if Point3d::are_points_collinear(&array_input[i..i + 4], tolerance) {
                        result = (i, i + 3);
                        break;
                    }
                }
                Some(result)
            } else {
                None
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

    // the following implementation for Point3d is when Point3d is use as ambiguous
    // representation of a Vector3d in order to avoid runtime penalty.
    // - using a Vertex will full fill the same purpose in a more idiomatic way.
    impl Point3d {
        /// Compute the magnitude of the point vector
        pub fn magnitude(&self) -> f64 {
            (self.X * self.X + self.Y * self.Y + self.Z * self.Z).sqrt()
        }

        /// Normalize the point as a vector
        /// (equivalent to unitize_b for Vector3d)
        /// this remove ambiguity when point 3d is use as vector
        /// ( to avoid sqrt penalty on magnetude creation when using Vector3d )
        /// - it's recommended to use Vertex for that.
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
        #[inline(always)]
        /// Convert a Point3d to Vertex.
        pub fn to_vertex(&self) -> Vertex {
            Vertex::new(self.X, self.Y, self.Z)
        }

        #[inline(always)]
        /// Return a standardized output.
        pub fn to_tuple(self) -> (f64, f64, f64) {
            (self.X, self.Y, self.Z)
        }

        #[inline(always)]
        /// Stabilize double precision number to a digit scale.
        ///  1e3 will round to 0.001
        ///  1e6 will rount to 0.000001
        pub fn clean_up_digits(&mut self, precision: f64) {
            self.X = self.X.trunc() + (self.X.fract() * precision).round() / precision;
            self.Y = self.Y.trunc() + (self.Y.fract() * precision).round() / precision;
            self.Z = self.Z.trunc() + (self.Z.fract() * precision).round() / precision;
        }
    }
    impl fmt::Display for Point3d {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "Point3d(x: {0:.3}, y: {1:.3}, z: {2:.3})",
                self.X, self.Y, self.Z
            )
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

    impl fmt::Display for Vector3d {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "Vector3d(x: {0:.3}, y: {1:.3}, z: {2:.3} Length:{3:.3})",
                self.X, self.Y, self.Z, self.Length
            )
        }
    }

    #[allow(non_snake_case)]
    impl Vector3d {
        #[inline(always)]
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

        #[inline(always)]
        pub fn set_X(&mut self, new_x_value: f64) {
            self.X = new_x_value;
            self.update_length();
        }

        #[inline(always)]
        pub fn set_Y(&mut self, new_y_value: f64) {
            self.Y = new_y_value;
            self.update_length();
        }

        #[inline(always)]
        pub fn set_Z(&mut self, new_z_value: f64) {
            self.Z = new_z_value;
            self.update_length();
        }

        #[inline(always)]
        pub fn get_X(&self) -> f64 {
            self.X
        }

        #[inline(always)]
        pub fn get_Y(&self) -> f64 {
            self.Y
        }

        #[inline(always)]
        pub fn get_Z(&self) -> f64 {
            self.Z
        }

        #[inline(always)]
        /// Stabilize double precision number to a digit scale.
        ///  1e3 will round to 0.001
        ///  1e6 will rount to 0.000001
        pub fn clean_up_digits(&mut self, precision: f64) {
            self.X = self.X.trunc() + (self.X.fract() * precision).round() / precision;
            self.Y = self.Y.trunc() + (self.Y.fract() * precision).round() / precision;
            self.Z = self.Z.trunc() + (self.Z.fract() * precision).round() / precision;
            self.update_length();
        }

        /// Return the read only length
        pub fn Length(&self) -> f64 {
            self.Length
        }

        /// Compute the vector length.
        fn update_length(&mut self) {
            self.Length = (self.X * self.X + self.Y * self.Y + self.Z * self.Z).sqrt();
        }

        /// Static way to compute vector length.
        /// # Arguments
        /// x:f 64, y:f 64, z:f 64
        /// # Returns
        /// Return a f 64 length distance.
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
        /// Normalize the vector from memory cached length.
        pub fn unitize(&mut self) {
            self.X /= self.Length;
            self.Y /= self.Length;
            self.Z /= self.Length;
            self.update_length();
        }

        #[inline(always)]
        /// Normalize the vector from memory cached length.
        pub fn unitize_b(&self) -> Self {
            let mag = self.Length;
            if mag > std::f64::EPSILON {
                Self {
                    X: self.X / mag,
                    Y: self.Y / mag,
                    Z: self.Z / mag,
                    Length: 1.0,
                }
            } else {
                Self {
                    X: 0.0,
                    Y: 0.0,
                    Z: 0.0,
                    Length: 0.0,
                }
            }
        }

        // the 3 following methods are alias for keeping
        // habits from other 3d Api.

        #[inline(always)]
        /// Same as unitize
        pub fn magnitude(&self) -> f64 {
            (self.X * self.X + self.Y * self.Y + self.Z * self.Z).sqrt()
        }

        #[inline(always)]
        /// Normalize the vector from memory cached length.
        pub fn normalize(&self) -> Self {
            let mag = self.Length;
            if mag > std::f64::EPSILON {
                Self {
                    X: self.X / mag,
                    Y: self.Y / mag,
                    Z: self.Z / mag,
                    Length: 1.0,
                }
            } else {
                Self {
                    X: 0.0,
                    Y: 0.0,
                    Z: 0.0,
                    Length: 0.0,
                }
            }
        }

        /// Compute the cross product of two vectors
        pub fn cross(&self, other: &Vector3d) -> Self {
            Vector3d::new(
                self.Y * other.Z - self.Z * other.Y,
                self.Z * other.X - self.X * other.Z,
                self.X * other.Y - self.Y * other.X,
            )
        }

        /// Test if a vector point to the direction of an other vector.
        /// # Arguments
        /// - ref &self,
        /// - other_vector:Vector3d (other vector to compare with),
        /// - threshold :f64
        /// (threshold should be closer to one for getting more precision. like: 0.99991)
        /// # Returns
        /// - true if looking in same direction or false if looking in other direction
        ///   (always from threshold value).
        pub fn is_same_direction(&self, other_vector: &Vector3d, threshold: f64) -> bool {
            if (*self).unitize_b() * (*other_vector).unitize_b() >= threshold {
                true
            } else {
                false
            }
        }

        /// Return the angle between two vectors
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

        /// Checks if a vector is (approximately) zero.
        pub fn is_zero(&self, tolerance: f64) -> bool {
            self.Length < tolerance
        }

        /// Deprecated use project on Cplane instead
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

        /// Project a vector on a CPlane.
        /// # Arguments
        ///   a CPlane as infinite plane representation.
        /// # Returns
        ///   a projected vector on the CPlane.
        pub fn project_on_cplane(&self, plane: &CPlane) -> Vector3d {
            (*self) - (plane.normal * ((*self) * plane.normal))
        }

        #[inline(always)]
        /// Return a standardized output.
        pub fn to_vertex(self) -> Vertex {
            Vertex::new(self.X, self.Y, self.Z)
        }

        #[inline(always)]
        /// Return a standardized output.
        pub fn to_tuple(self) -> (f64, f64, f64) {
            (self.X, self.Y, self.Z)
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

    impl Add<Point3d> for Vector3d {
        type Output = Point3d;
        fn add(self, point: Point3d) -> Self::Output {
            Point3d::new(self.X + point.X, self.Y + point.Y, self.Z + point.Z)
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

    /// Point3d, Vector3d, Vertex are bound to Coordinate3d.
    /// act as an interface for some functions.
    pub trait Coordinate3d {
        type Output;
        fn new(x: f64, y: f64, z: f64) -> Self::Output;
        fn get_x(&self) -> f64;
        fn get_y(&self) -> f64;
        fn get_z(&self) -> f64;
        fn to_tuple(&self) -> (f64, f64, f64);
    }
    impl Coordinate3d for Point3d {
        type Output = Point3d;
        fn new(x: f64, y: f64, z: f64) -> Self {
            Point3d::new(x, y, z)
        }
        fn get_x(&self) -> f64 {
            self.X
        }
        fn get_y(&self) -> f64 {
            self.Y
        }
        fn get_z(&self) -> f64 {
            self.Z
        }
        fn to_tuple(&self) -> (f64, f64, f64) {
            (self.X, self.Y, self.Z)
        }
    }

    impl Coordinate3d for Vector3d {
        type Output = Vector3d;
        fn new(x: f64, y: f64, z: f64) -> Self {
            Vector3d {
                X: x,
                Y: y,
                Z: z,
                Length: (x * x + y * y + z * z).sqrt(),
            }
        }
        fn get_x(&self) -> f64 {
            self.X
        }
        fn get_y(&self) -> f64 {
            self.Y
        }
        fn get_z(&self) -> f64 {
            self.Z
        }
        fn to_tuple(&self) -> (f64, f64, f64) {
            (self.X, self.Y, self.Z)
        }
    }

    impl Coordinate3d for Vertex {
        type Output = Vertex;
        fn new(x: f64, y: f64, z: f64) -> Self {
            Vertex::new(x, y, z)
        }
        fn get_x(&self) -> f64 {
            self.x
        }
        fn get_y(&self) -> f64 {
            self.y
        }
        fn get_z(&self) -> f64 {
            self.z
        }
        fn to_tuple(&self) -> (f64, f64, f64) {
            (self.x, self.y, self.z)
        }
    }

    /// Construction Plane tools set.
    /// - a full set of tools, for establishing a
    ///   reliable local coordinate system
    ///   from minimal input components.
    #[derive(Debug, Clone, Copy)]
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
        /// Construtor from input copy ideal for inline declaration but inputs are consumed.
        pub fn new_(origin: Point3d, normal: Vector3d) -> Self {
            let normalized_normal = normal.unitize_b();
            // Find a vector that is not parallel to the normal
            let mut arbitrary_vector = Vector3d::new(1.0, 0.0, 0.0);
            if normal.X.abs() > 0.99 {
                arbitrary_vector = Vector3d::new(0.0, 1.0, 0.0);
            }
            // Compute two orthogonal vectors on the plane using the cross product
            let u = Vector3d::cross_product(&normalized_normal, &arbitrary_vector).unitize_b();
            let v = Vector3d::cross_product(&normalized_normal, &u).unitize_b();
            Self {
                origin: Point3d::new(origin.X, origin.Y, origin.Z),
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
            let u = Vector3d::cross_product(&v, &normalized_normal).unitize_b();
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
            y_axis = -Vector3d::cross_product(&x_axis, &normal).unitize_b();
            Self {
                origin: *origin,
                normal,
                u: x_axis,
                v: y_axis,
            }
        }

        #[inline(always)]
        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        pub fn point_on_plane_uv(&self, u: f64, v: f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * u + self.v.X * v,
                Y: self.origin.Y + self.u.Y * u + self.v.Y * v,
                Z: self.origin.Z + self.u.Z * u + self.v.Z * v,
            }
        }

        #[inline(always)]
        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        /// Also offsets the point along the plane's normal by z value.
        pub fn point_on_plane(&self, x: f64, y: f64, z: f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * x + self.v.X * y + self.normal.X * z,
                Y: self.origin.Y + self.u.Y * x + self.v.Y * y + self.normal.Y * z,
                Z: self.origin.Z + self.u.Z * x + self.v.Z * y + self.normal.Z * z,
            }
        }

        #[inline(always)]
        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        pub fn point_on_plane_uv_ref(&self, u: &f64, v: &f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * (*u) + self.v.X * (*v),
                Y: self.origin.Y + self.u.Y * (*u) + self.v.Y * (*v),
                Z: self.origin.Z + self.u.Z * (*u) + self.v.Z * (*v),
            }
        }

        #[inline(always)]
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

    impl fmt::Display for CPlane {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let origin = format!(
                "Origin:(x:{0:0.3},y:{1:0.3},z:{2:0.3}) ",
                self.origin.X, self.origin.Y, self.origin.Z
            );
            let u = format!(
                "u vector:(x:{0:0.3},y:{1:0.3},z:{2:0.3}) ",
                self.u.X, self.u.Y, self.u.Z
            );
            let v = format!(
                "v vector:(x:{0:0.3},y:{1:0.3},z:{2:0.3}) ",
                self.v.X, self.v.Y, self.v.Z
            );
            let w = format!(
                "Normal vector:(x:{0:0.3},y:{1:0.3},z:{2:0.3}) ",
                self.normal.X, self.normal.Y, self.normal.Z
            );
            write!(f, "CPlane: {0},{1},{2},{3}", origin, u, v, w)
        }
    }

    /// Early implementation of an NurbsCurve.
    /// only basic functions like radius, curvature, degree and t evaluation
    /// are working ( cannot join curves  
    /// weight and knot system are partially implemented ).
    /// (i need to finish my evaluation system for going deeper conveniently)
    #[derive(Debug, Clone)]
    pub struct NurbsCurve {
        pub control_points: Vec<Point3d>,
        pub degree: usize,
        pub knots: Vec<f64>,
        pub weights: Vec<f64>,
    }

    impl NurbsCurve {
        /// Constructor for a NURBS curve.
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

        /// Compute the curvature of the NURBS curve at t parameter.
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
        /// Provide the radius of the oscultating circle at a t parameter.
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

        ///!!!!!! not working yet !!!!!!!!!!!!!! private method DONOT USE Patial implementation.
        /// Evaluate a NURBS curve at parameter `t` and return both the point and tangent vector
        fn evaluate_with_tangent(&self, t: f64) -> Option<(Point3d, Vector3d)> {
            let p = self.degree;

            // Ensure `t` is within the valid range
            if t < 0.0 || t > 1.0 {
                return None; // Invalid parameter
            }

            // Find the knot span
            let k = self.find_span(t);

            // Allocate arrays for de Boor points and weights
            let mut d = vec![
                Point3d {
                    X: 0.0,
                    Y: 0.0,
                    Z: 0.0
                };
                p + 1
            ];
            let mut w = vec![0.0; p + 1];

            // Initialize the de Boor points and weights
            for j in 0..=p {
                let cp = &self.control_points[k - p + j];
                let weight = self.weights[k - p + j];
                d[j] = Point3d {
                    X: cp.X * weight,
                    Y: cp.Y * weight,
                    Z: cp.Z * weight,
                };
                w[j] = weight;
            }

            // Perform the de Boor recursion for position
            for r in 1..=p {
                for j in (r..=p).rev() {
                    let alpha = (t - self.knots[k + j - p])
                        / (self.knots[k + 1 + j - r] - self.knots[k + j - p]);

                    d[j].X = (1.0 - alpha) * d[j - 1].X + alpha * d[j].X;
                    d[j].Y = (1.0 - alpha) * d[j - 1].Y + alpha * d[j].Y;
                    d[j].Z = (1.0 - alpha) * d[j - 1].Z + alpha * d[j].Z;

                    w[j] = (1.0 - alpha) * w[j - 1] + alpha * w[j];
                }
            }

            // Compute the position (normalize by weight)
            let point = Point3d {
                X: d[p].X / w[p],
                Y: d[p].Y / w[p],
                Z: d[p].Z / w[p],
            };

            // Tangent computation using first derivative of de Boor
            let mut d_tangent = vec![
                Point3d {
                    X: 0.0,
                    Y: 0.0,
                    Z: 0.0
                };
                p
            ];
            let mut w_tangent = vec![0.0; p];

            for j in 0..p {
                let weight_diff = self.weights[k - p + j + 1] - self.weights[k - p + j];
                d_tangent[j].X = (d[j + 1].X - d[j].X)
                    / (self.knots[k + j + 1] - self.knots[k + j])
                    + weight_diff * d[j].X;
                d_tangent[j].Y = (d[j + 1].Y - d[j].Y)
                    / (self.knots[k + j + 1] - self.knots[k + j])
                    + weight_diff * d[j].Y;
                d_tangent[j].Z = (d[j + 1].Z - d[j].Z)
                    / (self.knots[k + j + 1] - self.knots[k + j])
                    + weight_diff * d[j].Z;

                w_tangent[j] = (w[j + 1] - w[j]) / (self.knots[k + j + 1] - self.knots[k + j]);
            }

            // Perform the de Boor recursion for tangent
            for r in 1..p {
                for j in (r..p).rev() {
                    let alpha = (t - self.knots[k + j - p + 1])
                        / (self.knots[k + 1 + j - r] - self.knots[k + j - p + 1]);

                    d_tangent[j].X = (1.0 - alpha) * d_tangent[j - 1].X + alpha * d_tangent[j].X;
                    d_tangent[j].Y = (1.0 - alpha) * d_tangent[j - 1].Y + alpha * d_tangent[j].Y;
                    d_tangent[j].Z = (1.0 - alpha) * d_tangent[j - 1].Z + alpha * d_tangent[j].Z;

                    w_tangent[j] = (1.0 - alpha) * w_tangent[j - 1] + alpha * w_tangent[j];
                }
            }

            // Compute the tangent vector (normalize by weight derivative)
            let tangent = Vector3d::new(
                d_tangent[p - 1].X / w[p - 1],
                d_tangent[p - 1].Y / w[p - 1],
                d_tangent[p - 1].Z / w[p - 1],
            );

            Some((point, tangent))
        }

        /// !!! not finished yet !!! private method. do not use partial implementation.
        /// Evaluate a NURBS curve at parameter `t` and return:
        /// - The evaluated point
        /// - The tangent vector (1st derivative)
        /// - The acceleration vector (2nd derivative)
        fn evaluate_with_derivatives(&self, t: f64) -> Option<(Point3d, Vector3d, Vector3d)> {
            let p = self.degree;

            // Ensure `t` is within the valid range
            if t < 0.0 || t > 1.0 {
                return None; // Invalid parameter
            }

            // Find the knot span
            let k = self.find_span(t);

            // Allocate arrays for de Boor points and weights
            let mut d = vec![
                Point3d {
                    X: 0.0,
                    Y: 0.0,
                    Z: 0.0
                };
                p + 1
            ];
            let mut w = vec![0.0; p + 1];

            // Initialize the de Boor points and weights
            for j in 0..=p {
                let cp = &self.control_points[k - p + j];
                let weight = self.weights[k - p + j];
                d[j] = Point3d {
                    X: cp.X * weight,
                    Y: cp.Y * weight,
                    Z: cp.Z * weight,
                };
                w[j] = weight;
            }

            // Perform the de Boor recursion for position
            for r in 1..=p {
                for j in (r..=p).rev() {
                    let alpha = (t - self.knots[k + j - p])
                        / (self.knots[k + 1 + j - r] - self.knots[k + j - p]);

                    d[j].X = (1.0 - alpha) * d[j - 1].X + alpha * d[j].X;
                    d[j].Y = (1.0 - alpha) * d[j - 1].Y + alpha * d[j].Y;
                    d[j].Z = (1.0 - alpha) * d[j - 1].Z + alpha * d[j].Z;

                    w[j] = (1.0 - alpha) * w[j - 1] + alpha * w[j];
                }
            }

            // Compute the position (normalize by weight)
            let point = Point3d {
                X: d[p].X / w[p],
                Y: d[p].Y / w[p],
                Z: d[p].Z / w[p],
            };

            // Allocate arrays for 1st and 2nd derivative computations
            let mut d_tangent = vec![
                Point3d {
                    X: 0.0,
                    Y: 0.0,
                    Z: 0.0
                };
                p
            ];
            let mut w_tangent = vec![0.0; p];
            let mut d_acceleration = vec![
                Point3d {
                    X: 0.0,
                    Y: 0.0,
                    Z: 0.0
                };
                p - 1
            ];
            let mut w_acceleration = vec![0.0; p - 1];

            // Compute 1st derivatives (tangent vector)
            for j in 0..p {
                d_tangent[j].X =
                    (d[j + 1].X - d[j].X) / (self.knots[k + j + 1] - self.knots[k + j]);
                d_tangent[j].Y =
                    (d[j + 1].Y - d[j].Y) / (self.knots[k + j + 1] - self.knots[k + j]);
                d_tangent[j].Z =
                    (d[j + 1].Z - d[j].Z) / (self.knots[k + j + 1] - self.knots[k + j]);

                w_tangent[j] = (w[j + 1] - w[j]) / (self.knots[k + j + 1] - self.knots[k + j]);
            }

            // Perform de Boor recursion for 1st derivatives
            for r in 1..p {
                for j in (r..p).rev() {
                    let alpha = (t - self.knots[k + j - p + 1])
                        / (self.knots[k + 1 + j - r] - self.knots[k + j - p + 1]);

                    d_tangent[j].X = (1.0 - alpha) * d_tangent[j - 1].X + alpha * d_tangent[j].X;
                    d_tangent[j].Y = (1.0 - alpha) * d_tangent[j - 1].Y + alpha * d_tangent[j].Y;
                    d_tangent[j].Z = (1.0 - alpha) * d_tangent[j - 1].Z + alpha * d_tangent[j].Z;

                    w_tangent[j] = (1.0 - alpha) * w_tangent[j - 1] + alpha * w_tangent[j];
                }
            }

            // Compute 2nd derivatives (acceleration vector)
            for j in 0..p - 1 {
                d_acceleration[j].X = (d_tangent[j + 1].X - d_tangent[j].X)
                    / (self.knots[k + j + 2] - self.knots[k + j + 1]);
                d_acceleration[j].Y = (d_tangent[j + 1].Y - d_tangent[j].Y)
                    / (self.knots[k + j + 2] - self.knots[k + j + 1]);
                d_acceleration[j].Z = (d_tangent[j + 1].Z - d_tangent[j].Z)
                    / (self.knots[k + j + 2] - self.knots[k + j + 1]);

                w_acceleration[j] = (w_tangent[j + 1] - w_tangent[j])
                    / (self.knots[k + j + 2] - self.knots[k + j + 1]);
            }

            // Perform de Boor recursion for 2nd derivatives
            for r in 1..p - 1 {
                for j in (r..p - 1).rev() {
                    let alpha = (t - self.knots[k + j - p + 2])
                        / (self.knots[k + 1 + j - r] - self.knots[k + j - p + 2]);

                    d_acceleration[j].X =
                        (1.0 - alpha) * d_acceleration[j - 1].X + alpha * d_acceleration[j].X;
                    d_acceleration[j].Y =
                        (1.0 - alpha) * d_acceleration[j - 1].Y + alpha * d_acceleration[j].Y;
                    d_acceleration[j].Z =
                        (1.0 - alpha) * d_acceleration[j - 1].Z + alpha * d_acceleration[j].Z;

                    w_acceleration[j] =
                        (1.0 - alpha) * w_acceleration[j - 1] + alpha * w_acceleration[j];
                }
            }

            // Normalize tangent and acceleration by respective weights
            let tangent = Vector3d::new(
                d_tangent[p - 1].X / w[p - 1],
                d_tangent[p - 1].Y / w[p - 1],
                d_tangent[p - 1].Z / w[p - 1],
            );

            let acceleration = Vector3d::new(
                d_acceleration[p - 2].X / w[p - 2],
                d_acceleration[p - 2].Y / w[p - 2],
                d_acceleration[p - 2].Z / w[p - 2],
            );

            Some((point, tangent, acceleration))
        }

        /// Describe the path by a list of Point3d.
        pub fn describe_path(&self, step_resolution: f64) -> Vec<Point3d> {
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

        /// working for now that the one to use.
        pub fn numerical_first_derivative(&self, t: f64, h: f64) -> Point3d {
            let pt_plus_h = self.evaluate(t + h);
            let pt_minus_h = self.evaluate(t - h);
            Point3d::new(
                (pt_plus_h.X - pt_minus_h.X) / (2.0 * h),
                (pt_plus_h.Y - pt_minus_h.Y) / (2.0 * h),
                (pt_plus_h.Z - pt_minus_h.Z) / (2.0 * h),
            )
        }
        /// working for now that the one to use.
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
        /// Constructor like with OpenNurbs Lib.
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

    #[allow(non_snake_case)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Vector2d {
        pub X: f64,
        pub Y: f64,
    }
    impl Vector2d {
        pub fn new(x: f64, y: f64) -> Self {
            Self { X: x, Y: y }
        }
        pub fn magnetude(self) -> f64 {
            (self.X * self.X + self.Y * self.Y).sqrt()
        }
        pub fn normalize(self) -> Self {
            let m = self.magnetude();
            if m > std::f64::EPSILON {
                Self {
                    X: self.X / m,
                    Y: self.Y / m,
                }
            } else {
                Self { X: 0.0, Y: 0.0 }
            }
        }
        pub fn crossproduct(first_vector: &Vector2d, second_vector: &Vector2d) -> f64 {
            ((*first_vector).X * (*second_vector).Y) - ((*first_vector).Y * (*second_vector).X)
        }
        pub fn angle(first_vector: &Vector2d, second_vector: &Vector2d) -> f64 {
            f64::acos(
                ((*first_vector) * (*second_vector))
                    / ((*first_vector).magnetude() * (*second_vector).magnetude()),
            )
        }
        pub fn to_tuple(&self) -> (f64, f64) {
            (self.X, self.Y)
        }
    }

    impl fmt::Display for Vector2d {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Vector2d{{x:{0},y:{1}}}", self.X, self.Y)
        }
    }
    impl Sub for Vector2d {
        type Output = Self;
        fn sub(self, other: Vector2d) -> Self::Output {
            Self {
                X: self.X - other.X,
                Y: self.Y - other.Y,
            }
        }
    }

    impl Add for Vector2d {
        type Output = Self;
        fn add(self, other: Vector2d) -> Self::Output {
            Self {
                X: self.X + other.X,
                Y: self.Y + other.Y,
            }
        }
    }
    impl Mul<f64> for Vector2d {
        type Output = Self;
        fn mul(self, scalar: f64) -> Self::Output {
            Self {
                X: self.X * scalar,
                Y: self.Y * scalar,
            }
        }
    }
    impl Mul for Vector2d {
        type Output = f64;
        fn mul(self, other: Vector2d) -> f64 {
            self.X * other.X + self.Y * other.Y
        }
    }
    impl Div for Vector2d {
        type Output = Self;
        fn div(self, other: Vector2d) -> Self::Output {
            Self {
                X: self.X / other.X,
                Y: self.Y / other.Y,
            }
        }
    }
    impl MulAssign<f64> for Vector2d {
        fn mul_assign(&mut self, scalar: f64) {
            self.X *= scalar;
            self.Y *= scalar;
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
        /// Computes the dot product of two quaternions
        pub fn dot(self, other: Quaternion) -> f64 {
            self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
        }

        /// Performs SLERP (Spherical Linear Interpolation) between two quaternions
        pub fn slerp(start: Quaternion, end: Quaternion, t: f64) -> Quaternion {
            // Normalize the quaternions
            let start = start.normalize();
            let mut end = end.normalize();

            // Compute the dot product
            let mut dot = start.dot(end);

            // If the dot product is negative, invert one quaternion to take the shortest path
            if dot < 0.0 {
                end = Quaternion::new(-end.w, -end.x, -end.y, -end.z);
                dot = -dot;
            }

            // If the quaternions are very close, use linear interpolation (LERP) as a fallback
            if dot > 0.9995 {
                return Quaternion::new(
                    start.w + t * (end.w - start.w),
                    start.x + t * (end.x - start.x),
                    start.y + t * (end.y - start.y),
                    start.z + t * (end.z - start.z),
                )
                .normalize();
            }

            // Compute the angle theta between the quaternions
            let theta_0 = dot.acos(); // Original angle
            let theta = theta_0 * t; // Interpolated angle

            // Compute sin values
            let sin_theta = theta.sin();
            let sin_theta_0 = theta_0.sin();

            // Interpolation factors
            let s1 = (1.0 - t) * sin_theta / sin_theta_0;
            let s2 = t * sin_theta / sin_theta_0;

            // Perform SLERP
            Quaternion::new(
                s1 * start.w + s2 * end.w,
                s1 * start.x + s2 * end.x,
                s1 * start.y + s2 * end.y,
                s1 * start.z + s2 * end.z,
            )
            .normalize()
        }

        // Rotate a point using the quaternion
        pub fn rotate_point<T: Coordinate3d>(&self, point: &T) -> T
        where
            T: Coordinate3d<Output = T>,
        {
            let q_point = Quaternion::new(0.0, point.get_x(), point.get_y(), point.get_z());
            let q_conjugate = self.conjugate();
            let rotated_q = self.multiply(&q_point).multiply(&q_conjugate);
            T::new(rotated_q.x, rotated_q.y, rotated_q.z)
        }

        pub fn rotate_point_around_axis<T: Coordinate3d>(point: &T, axis: &T, angle: f64) -> T
        where
            T: Coordinate3d<Output = T>,
            //let pt1 = camera.project_maybe_outside(&local_x);
        {
            // Normalize the axis vector
            let axis_length = (axis.get_x() * axis.get_x()
                + axis.get_y() * axis.get_y()
                + axis.get_z() * axis.get_z())
            .sqrt();
            let axis_normalized = Vertex::new(
                axis.get_x() / axis_length,
                axis.get_y() / axis_length,
                axis.get_z() / axis_length,
            );

            // Create the quaternion for the rotation
            let half_angle = angle.to_radians() / 2.0;
            let sin_half_angle = half_angle.sin();
            let rotation_quat = Quaternion::new(
                half_angle.cos(),
                axis_normalized.x * sin_half_angle,
                axis_normalized.y * sin_half_angle,
                axis_normalized.z * sin_half_angle,
            )
            .normalize();
            // Rotate the point using the quaternion
            rotation_quat.rotate_point(point)
        }

        pub fn rotate_point_around_axis_to_4x4(axis: &Vertex, angle: f64) -> [[f64; 4]; 4] {
            // Normalize the axis vector
            let axis = axis.normalize();
            let angle_rad = angle.to_radians();
            // Create the quaternion for the rotation
            let half_angle = angle_rad / 2.0;
            let sin_half_angle = half_angle.sin();
            let rotation_quat = Quaternion::new(
                half_angle.cos(),
                axis.x * sin_half_angle,
                axis.y * sin_half_angle,
                axis.z * sin_half_angle,
            )
            .normalize();
            // produce a 4x4 matrix.
            rotation_quat.to_4x4_matrix()
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
        pub fn to_4x3_matrix(&self) -> [[f64; 4]; 3] {
            let (w, x, y, z) = (self.w, self.x, self.y, self.z);
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                    0.0, // Translation x part left at zero
                         // (quaternion represent only rotation.)
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                    0.0, // Translation y part left at zero.
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                    0.0, // Translation z part left at zero.
                ],
            ]
        }
    }
}

// Everything related to intersections or Zones evaluation.
pub mod intersection {
    use super::geometry::{CPlane, Point3d, Vector3d};
    // Act as an interface "the object can hold the
    // Circle or the Rectangle type inside a same collection list.
    pub enum ClickZoneType {
        ClickZoneCircle(Circle),
        ClickZoneRectangle(Rectangle),
    }
    pub struct Rectangle {
        pub pt1: (usize, usize),
        pub pt2: (usize, usize),
    }
    impl Rectangle {
        /// Create a rectangle zone area from 2 points
        /// describing it's diagonal start and end point.
        pub fn new(first_point: (usize, usize), second_point: (usize, usize)) -> Self {
            Self {
                pt1: first_point,
                pt2: second_point,
            }
        }
        /// Test if a point is inside the rectangle zone.
        pub fn is_point_inside(self, test_point: (usize, usize)) -> bool {
            if (test_point.0 >= self.pt1.0)
                && (test_point.1 >= self.pt1.1)
                && (test_point.0 <= self.pt2.0)
                && (test_point.1 <= self.pt2.1)
            {
                true
            } else {
                false
            }
        }
    }
    /// Construct a circle zone area.
    pub struct Circle {
        center_point: (usize, usize),
        radius: f64,
    }

    impl Circle {
        /// Create a circle zone area from it's center point and radius.
        pub fn new(center_point: (usize, usize), radius: f64) -> Self {
            Self {
                center_point,
                radius,
            }
        }

        /// Test if a point is inside the circle zone area.
        pub fn is_point_inside(self, test_point: (usize, usize)) -> bool {
            // Compute distance from circle center.
            let dx = self.center_point.0 as isize - test_point.0 as isize;
            let dy = self.center_point.1 as isize - test_point.1 as isize;
            let squared_ditance = (dx as f64) * (dx as f64) + (dy as f64) * (dy as f64);
            // println!("\x1b[1;1HDistance from circle center: {0:?}",squared_ditance.sqrt());
            // Check if the squared circle is inside the circle.
            if squared_ditance.sqrt() <= self.radius {
                true
            } else {
                false
            }
        }
    }

    /// Project a Point3d on a CPlane through a ray vector.
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

    /// - an Early function deprecated.
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
        if !Vector3d::are_coplanar_a(d1, d2) {
            None // if lines never intersect.
        } else {
            let diff = *p2 - *p1; // Make vector delta.
            let t1 = Vector3d::cross_product(&diff, d2) * cross_d1_d2 / denom; // Compute intersection from formula.
            Some(*p1 + ((*d1) * t1)) // Return result.
        }
    }

    /// Cohen-Sutherland Line Clipping Algorithm
    pub fn clip_line(
        mut pt1: (f64, f64),
        mut pt2: (f64, f64),
        screen_width: usize,
        screen_height: usize,
    ) -> Option<((f64, f64), (f64, f64))> {
        // Screen boundaries
        let xmin = 0.0;
        let ymin = 0.0;
        let xmax = (screen_width - 1) as f64;
        let ymax = (screen_height - 1) as f64;

        // Region codes
        const INSIDE: u8 = 0; // 0000
        const LEFT: u8 = 1; // 0001
        const RIGHT: u8 = 2; // 0010
        const BOTTOM: u8 = 4; // 0100
        const TOP: u8 = 8; // 1000

        // Function to compute the region code for a point
        fn compute_code(x: f64, y: f64, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> u8 {
            let mut code = INSIDE;
            if x < xmin {
                code |= LEFT;
            } else if x > xmax {
                code |= RIGHT;
            }
            if y < ymin {
                code |= BOTTOM;
            } else if y > ymax {
                code |= TOP;
            }
            code
        }

        let mut code1 = compute_code(pt1.0, pt1.1, xmin, ymin, xmax, ymax);
        let mut code2 = compute_code(pt2.0, pt2.1, xmin, ymin, xmax, ymax);

        while code1 != 0 || code2 != 0 {
            // If both points are outside the same region, the line is completely outside
            if code1 & code2 != 0 {
                return None;
            }

            // Choose the point outside the screen to clip
            let (code_out, x, y);
            if code1 != 0 {
                code_out = code1;
                x = pt1.0;
                y = pt1.1;
            } else {
                code_out = code2;
                x = pt2.0;
                y = pt2.1;
            }

            // Find intersection with the screen boundary
            let (new_x, new_y) = if code_out & TOP != 0 {
                // Clip to the top edge
                (x + (pt2.0 - pt1.0) * (ymax - y) / (pt2.1 - pt1.1), ymax)
            } else if code_out & BOTTOM != 0 {
                // Clip to the bottom edge
                (x + (pt2.0 - pt1.0) * (ymin - y) / (pt2.1 - pt1.1), ymin)
            } else if code_out & RIGHT != 0 {
                // Clip to the right edge
                (xmax, y + (pt2.1 - pt1.1) * (xmax - x) / (pt2.0 - pt1.0))
            } else if code_out & LEFT != 0 {
                // Clip to the left edge
                (xmin, y + (pt2.1 - pt1.1) * (xmin - x) / (pt2.0 - pt1.0))
            } else {
                unreachable!()
            };

            // Update the point and its region code
            if code_out == code1 {
                pt1 = (new_x, new_y);
                code1 = compute_code(pt1.0, pt1.1, xmin, ymin, xmax, ymax);
            } else {
                pt2 = (new_x, new_y);
                code2 = compute_code(pt2.0, pt2.1, xmin, ymin, xmax, ymax);
            }
        }

        Some((pt1, pt2))
    }
}

pub mod transformation {
    use super::geometry::Coordinate3d;
    use super::geometry::Point3d;
    use rayon::prelude::*;

    // first basic Euler transformation.////////////////////////////////
    /// Rotate the point from X world axis (Euler rotation).
    /// # Arguments
    /// input T generic can be Point3d, Vector3d or a Vertex type as input.
    /// # Returns
    /// return a Point3d Vector3d or a Vertex depend on input type.
    pub fn rotate_x<T: Coordinate3d>(point: T, angle: f64) -> T
    where
        T: Coordinate3d<Output = T>,
    {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        T::new(
            point.get_x(),
            point.get_y() * cos_theta - point.get_z() * sin_theta,
            point.get_y() * sin_theta + point.get_z() * cos_theta,
        )
    }

    /// Rotate the point from Y world axis (Euler rotation).
    /// # Arguments
    /// !!! the T generic can be Point3d, Vector3d or a Vertex type as input.
    /// # Returns
    /// return a Point3d Vector3d or a Vertex depend on input type.
    pub fn rotate_y<T: Coordinate3d>(point: T, angle: f64) -> T
    where
        T: Coordinate3d<Output = T>,
    {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        T::new(
            point.get_x() * cos_theta - point.get_z() * sin_theta,
            point.get_y(),
            point.get_x() * sin_theta + point.get_z() * cos_theta,
        )
    }

    /// Rotate the point from X world axis (Euler rotation).
    /// # Arguments
    /// !!! the T generic can be Point3d, Vector3d or a Vertex type as input.
    /// # Returns
    /// return a Point3d Vector3d or a Vertex depend on input type.
    pub fn rotate_z<T: Coordinate3d>(point: T, angle: f64) -> T
    where
        T: Coordinate3d<Output = T>,
    {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        T::new(
            point.get_x() * cos_theta - point.get_y() * sin_theta,
            point.get_x() * sin_theta + point.get_y() * cos_theta,
            point.get_z(),
        )
    }
    ////////////////////////////////////////////////////////////////////////////
    // 4x4 Matrix transformation part.
    // following a set of methods for generating specific transformation matrix
    // then some tools for being combined in to one.
    // and a method to process the transformation to a stack of Coordinate3d.
    ////////////////////////////////////////////////////////////////////////////
    /// Create a translation from a Point3d a Vector3d or a Vertex (interpreted as a vector)
    pub fn translation_matrix_from_vector<T: Coordinate3d>(translation_vector: T) -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, translation_vector.get_x()],
            [0.0, 1.0, 0.0, translation_vector.get_y()],
            [0.0, 0.0, 1.0, translation_vector.get_z()],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Make a scale matrix from an arbitrary center and x y z scale ratio.  
    pub fn scaling_matrix_from_center<T: Coordinate3d>(
        center: T,
        scale_x: f64,
        scale_y: f64,
        scale_z: f64,
    ) -> [[f64; 4]; 4] {
        // Extract the center coordinates
        let cx = center.get_x();
        let cy = center.get_y();
        let cz = center.get_z();

        // Translation to the origin
        let translation_to_origin = [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 1.0, -cz],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Scaling matrix
        let scaling = [
            [scale_x, 0.0, 0.0, 0.0],
            [0.0, scale_y, 0.0, 0.0],
            [0.0, 0.0, scale_z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Translation back to the original center
        let translation_back = [
            [1.0, 0.0, 0.0, cx],
            [0.0, 1.0, 0.0, cy],
            [0.0, 0.0, 1.0, cz],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Combine the matrices: T_back * S * T_to_origin
        let temp_matrix = multiply_matrices_axb(scaling, translation_to_origin);
        let final_matrix = multiply_matrices_axb(translation_back, temp_matrix);

        final_matrix
    }

    /// Make a scale matrix from world origin and x y z scale ratio.
    pub fn scale_matrix_from_ratio_xyz(
        scale_ratio_x: f64,
        scale_ratio_y: f64,
        scale_ratio_z: f64,
    ) -> [[f64; 4]; 4] {
        [
            [scale_ratio_x, 0.0, 0.0, 0.0],
            [0.0, scale_ratio_y, 0.0, 0.0],
            [0.0, 0.0, scale_ratio_z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
    /// Create a rotation matrix from angle (in degrees) for X
    pub fn rotation_matrix_on_x(x_angle: f64) -> [[f64; 4]; 4] {
        // Convert angles from degrees to radians
        let x_rad = x_angle.to_radians();
        // Rotation matrix around the X-axis
        let rotation_x = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, x_rad.cos(), -x_rad.sin(), 0.0],
            [0.0, x_rad.sin(), x_rad.cos(), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        rotation_x
    }

    /// Create a rotation matrix from angle (in degrees) for Y
    pub fn rotation_matrix_on_y(y_angle: f64) -> [[f64; 4]; 4] {
        // Convert angles from degrees to radians
        let y_rad = y_angle.to_radians();
        // Rotation matrix around the Y-axis
        let rotation_y = [
            [y_rad.cos(), 0.0, y_rad.sin(), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-y_rad.sin(), 0.0, y_rad.cos(), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        rotation_y
    }

    /// Create a rotation matrix from angle (in degrees) for Z
    pub fn rotation_matrix_on_z(z_angle: f64) -> [[f64; 4]; 4] {
        // Convert angles from degrees to radians
        let z_rad = z_angle.to_radians();
        // Rotation matrix around the Z-axis
        let rotation_z = [
            [z_rad.cos(), -z_rad.sin(), 0.0, 0.0],
            [z_rad.sin(), z_rad.cos(), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        rotation_z
    }

    /// Create a rotation matrix from angles (in degrees) for X, Y, and Z axis
    pub fn rotation_matrix_from_angles(x_angle: f64, y_angle: f64, z_angle: f64) -> [[f64; 4]; 4] {
        // Convert angles from degrees to radians
        let x_rad = x_angle.to_radians();
        let y_rad = y_angle.to_radians();
        let z_rad = z_angle.to_radians();

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

        // Rotation matrix around the Z-axis
        let rotation_z = [
            [z_rad.cos(), -z_rad.sin(), 0.0, 0.0],
            [z_rad.sin(), z_rad.cos(), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Combine the rotation matrices: R = Rz * Ry * Rx
        let rotation_xy = multiply_matrices_axb(rotation_y, rotation_x);
        let rotation_xyz = multiply_matrices_axb(rotation_z, rotation_xy);

        rotation_xyz
    }

    /// Rotate a matrix from an angle and a vector axis.
    pub fn rotation_matrix_from_axis_angle<T: Coordinate3d>(axis: T, angle: f64) -> [[f64; 4]; 4] {
        // Normalize the axis vector
        let x = axis.get_x();
        let y = axis.get_y();
        let z = axis.get_z();
        let length = (x * x + y * y + z * z).sqrt();

        if length == 0.0 {
            panic!("Axis vector cannot be zero-length.");
        }

        let ux = x / length;
        let uy = y / length;
        let uz = z / length;

        // Convert the angle to radians
        let theta = angle.to_radians();

        // Precompute trigonometric terms
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let one_minus_cos = 1.0 - cos_theta;

        // Compute the rotation matrix using the axis-angle formula
        [
            [
                cos_theta + ux * ux * one_minus_cos,
                ux * uy * one_minus_cos - uz * sin_theta,
                ux * uz * one_minus_cos + uy * sin_theta,
                0.0,
            ],
            [
                uy * ux * one_minus_cos + uz * sin_theta,
                cos_theta + uy * uy * one_minus_cos,
                uy * uz * one_minus_cos - ux * sin_theta,
                0.0,
            ],
            [
                uz * ux * one_minus_cos - uy * sin_theta,
                uz * uy * one_minus_cos + ux * sin_theta,
                cos_theta + uz * uz * one_minus_cos,
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Combine multiple transformation matrix into one...
    /// since transformation order is not commutative order history must be kept
    /// so a function call from stack queue is required use the macro vec! for that.
    pub fn combine_matrices(matrices: Vec<[[f64; 4]; 4]>) -> [[f64; 4]; 4] {
        // Start with the identity matrix
        let mut result = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Multiply all matrices together in sequence
        for matrix in matrices {
            result = multiply_matrices_axb(result, matrix);
        }

        result
    }

    /// Helper function to multiply two 4x4 matrices
    pub fn multiply_matrices_axb(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] =
                    a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
            }
        }
        result
    }

    /// Apply a transformations matrix to a point
    /// input data can be Point3d Vector3d or a Vertex
    pub fn transform_point<T: Coordinate3d + Send + Sync>(
        tranform_matrix: &[[f64; 4]; 4],
        point_to_process: &T,
    ) -> T
    where
        T: Coordinate3d<Output = T>,
    {
        T::new(
            tranform_matrix[0][0] * (*point_to_process).get_x()
                + tranform_matrix[0][1] * (*point_to_process).get_y()
                + tranform_matrix[0][2] * (*point_to_process).get_z()
                + tranform_matrix[0][3] * 1.0,
            tranform_matrix[1][0] * (*point_to_process).get_x()
                + tranform_matrix[1][1] * (*point_to_process).get_y()
                + tranform_matrix[1][2] * (*point_to_process).get_z()
                + tranform_matrix[1][3] * 1.0,
            tranform_matrix[2][0] * (*point_to_process).get_x()
                + tranform_matrix[2][1] * (*point_to_process).get_y()
                + tranform_matrix[2][2] * (*point_to_process).get_z()
                + tranform_matrix[2][3] * 1.0,
        )
    }

    /// Apply a transformations matrix to a list of points
    /// input data can be Point3d Vector3d or a Vertex
    pub fn transform_points<T: Coordinate3d + Send + Sync>(
        tranform_matrix: &[[f64; 4]; 4],
        points_to_process: &Vec<T>,
    ) -> Vec<T>
    where
        T: Coordinate3d<Output = T>,
    {
        points_to_process
            .par_iter()
            .map(|point| {
                T::new(
                    tranform_matrix[0][0] * (*point).get_x()
                        + tranform_matrix[0][1] * (*point).get_y()
                        + tranform_matrix[0][2] * (*point).get_z()
                        + tranform_matrix[0][3] * 1.0,
                    tranform_matrix[1][0] * (*point).get_x()
                        + tranform_matrix[1][1] * (*point).get_y()
                        + tranform_matrix[1][2] * (*point).get_z()
                        + tranform_matrix[1][3] * 1.0,
                    tranform_matrix[2][0] * (*point).get_x()
                        + tranform_matrix[2][1] * (*point).get_y()
                        + tranform_matrix[2][2] * (*point).get_z()
                        + tranform_matrix[2][3] * 1.0,
                )
            })
            .collect()
    }

    /// Create a 4x3 rotation matrix from angles (in degrees) for X, Y, and Z axes
    pub fn rotation_matrix_from_angles_4x3(
        x_angle: f64,
        y_angle: f64,
        z_angle: f64,
    ) -> [[f64; 4]; 3] {
        // Convert angles from degrees to radians
        let x_rad = x_angle.to_radians();
        let y_rad = y_angle.to_radians();
        let z_rad = z_angle.to_radians();

        // Rotation matrix around the X-axis
        let rotation_x = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, x_rad.cos(), -x_rad.sin(), 0.0],
            [0.0, x_rad.sin(), x_rad.cos(), 0.0],
        ];

        // Rotation matrix around the Y-axis
        let rotation_y = [
            [y_rad.cos(), 0.0, y_rad.sin(), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-y_rad.sin(), 0.0, y_rad.cos(), 0.0],
        ];

        // Rotation matrix around the Z-axis
        let rotation_z = [
            [z_rad.cos(), -z_rad.sin(), 0.0, 0.0],
            [z_rad.sin(), z_rad.cos(), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];

        // Combine the rotation matrices: R = Rz * Ry * Rx
        let rotation_xy = multiply_matrices_4x3(rotation_y, rotation_x);
        let rotation_xyz = multiply_matrices_4x3(rotation_z, rotation_xy);

        rotation_xyz
    }

    /// Generate a 4x3 translation matrix from a translation vector
    pub fn translation_matrix_4x3<T: Coordinate3d>(translation: T) -> [[f64; 4]; 3] {
        [
            [1.0, 0.0, 0.0, translation.get_x()],
            [0.0, 1.0, 0.0, translation.get_y()],
            [0.0, 0.0, 1.0, translation.get_z()],
        ]
    }

    /// Helper function to multiply two 4x3 matrices
    pub fn multiply_matrices_4x3(a: [[f64; 4]; 3], b: [[f64; 4]; 3]) -> [[f64; 4]; 3] {
        let mut result = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..4 {
                result[i][j] = a[i][0] * b[0][j]
                    + a[i][1] * b[1][j]
                    + a[i][2] * b[2][j]
                    + if j == 3 { a[i][3] } else { 0.0 };
            }
        }
        result
    }

    /// Combine multiple 4x3 transformation matrices into one
    pub fn combine_matrices_4x3(matrices: Vec<[[f64; 4]; 3]>) -> [[f64; 4]; 3] {
        let mut result = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        for matrix in matrices {
            result = multiply_matrices_4x3(result, matrix);
        }
        result
    }

    /// Create a 4x3 matix to scale input points relative to a center point
    pub fn scaling_matrix_from_center_4x3<T: Coordinate3d>(
        center: T,
        scale_x: f64,
        scale_y: f64,
        scale_z: f64,
    ) -> [[f64; 4]; 3] {
        // Extract the center coordinates
        let cx = center.get_x();
        let cy = center.get_y();
        let cz = center.get_z();

        // Compute the translation components after scaling
        let translation_x = cx - (scale_x * cx);
        let translation_y = cy - (scale_y * cy);
        let translation_z = cz - (scale_z * cz);

        // Construct the 4x3 matrix with [[f64; 4]; 3] format
        [
            [scale_x, 0.0, 0.0, translation_x], // Row 1: scale_x and translation_x
            [0.0, scale_y, 0.0, translation_y], // Row 2: scale_y and translation_y
            [0.0, 0.0, scale_z, translation_z], // Row 3: scale_z and translation_z
        ]
    }

    /// Apply a 4x3 transformation matrix to a vector of points
    pub fn transform_points_4x3<T: Coordinate3d + Send + Sync>(
        transform_matrix: &[[f64; 4]; 3],
        points_to_process: &Vec<T>,
    ) -> Vec<T>
    where
        T: Coordinate3d<Output = T>,
    {
        points_to_process
            .par_iter()
            .map(|point| {
                T::new(
                    transform_matrix[0][0] * point.get_x()
                        + transform_matrix[0][1] * point.get_y()
                        + transform_matrix[0][2] * point.get_z()
                        + transform_matrix[0][3],
                    transform_matrix[1][0] * point.get_x()
                        + transform_matrix[1][1] * point.get_y()
                        + transform_matrix[1][2] * point.get_z()
                        + transform_matrix[1][3],
                    transform_matrix[2][0] * point.get_x()
                        + transform_matrix[2][1] * point.get_y()
                        + transform_matrix[2][2] * point.get_z()
                        + transform_matrix[2][3],
                )
            })
            .collect()
    }

    #[inline(always)]
    /// Apply a 4x3 transformation matrix to a
    /// points of implementing Coordinate3d.
    pub fn transform_point_4x3<T: Coordinate3d + Send + Sync>(
        transform_matrix: &[[f64; 4]; 3],
        point_to_process: &T,
    ) -> T
    where
        T: Coordinate3d<Output = T>,
    {
        T::new(
            transform_matrix[0][0] * point_to_process.get_x()
                + transform_matrix[0][1] * point_to_process.get_y()
                + transform_matrix[0][2] * point_to_process.get_z()
                + transform_matrix[0][3],
            transform_matrix[1][0] * point_to_process.get_x()
                + transform_matrix[1][1] * point_to_process.get_y()
                + transform_matrix[1][2] * point_to_process.get_z()
                + transform_matrix[1][3],
            transform_matrix[2][0] * point_to_process.get_x()
                + transform_matrix[2][1] * point_to_process.get_y()
                + transform_matrix[2][2] * point_to_process.get_z()
                + transform_matrix[2][3],
        )
    }

    /// Deprecated. use Vector3d method to project on CPlane instead.
    /// Project a 3d point on a 4 defined Point3d plane (from the plane Vector Normal)
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

//TODO: the Draw method of a NurbsCurve must be in draw module.
pub mod draw {
    use super::geometry::CPlane;
    use super::geometry::Point3d;
    use super::utillity;
    use crate::fonts_txt::FONT_5X7;
    use crate::render_tools::rendering_object::Vertex;
    use crate::render_tools::visualization_v4::Camera;
    use core::f64;
    use std::usize;

    /// Draw a line very fast without antialiasing.
    /// Bresenham's line algorithm.
    /// Draw a line between two 2d points on screen.
    /// it's a clever algorithm dynamically plotting
    /// the distance between two points.
    /// - Bresenham's algorithm compute at each loop the direction (x,y)
    ///   of the next 2d point to plot and so, draw a line in
    ///   a (x,y) space and efficiently create the illusion of
    ///   a 3d line moving or rotating
    ///   (if created with a 3d point projected in 2d).
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

    /// Xiaolin Wu's line algorithm
    /// A good function for drawing clean
    /// anti-aliased line.
    pub fn draw_aa_line_with_thickness(
        buffer: &mut Vec<u32>,
        screen_width: usize,
        mut pt1: (f64, f64),
        mut pt2: (f64, f64),
        thickness: usize,
        color: u32,
    ) {
        let screen_height = buffer.len() / screen_width;
        let half_thickness = thickness / 2;
        ///////////////////////////////////////////////////////////////////////
        if (pt2.1 - pt1.1).abs() < (pt2.0 - pt1.0).abs() {
            // Swap x and y for writing line from end
            // to start when pt2.x is inferior to pt1.x
            if pt2.0 < pt1.0 {
                (pt1, pt2) = (pt2, pt1);
            }
            // Compute end point distances on x and y.
            let dx = pt2.0 - pt1.0;
            let dy = pt2.1 - pt1.1;
            //////////////////////////////////////////
            // Avoid division by zero for
            // defining the slope ration m.
            let m = if dx != 0.0 {
                dy / dx // compute the slope ratio.
            } else {
                dy / 1.0 // slope if dx == 0 then replace by 1.
            };
            // From 2nd point to penultimate point.
            for i in 0..=(dx.ceil() as usize) {
                // Move x px from + i on x axis
                let frac_x = pt1.0 + (i as f64);
                // Move the y px from slope ratio time
                // the n iteration step as scalar factor.
                let frac_y = pt1.1 + (i as f64) * m;
                // Convert x and y in integer.
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_y - (y as f64); // Get only the fractional part.
                for j in 0..=thickness {
                    let y_offset = (y + j).saturating_sub(half_thickness) + 1; // problematic part.
                                                                               // panic!("----->{y_offset} x:{x} y:{y} j:{j} half_thickness{half_thickness}");
                    if (x < screen_width) && (y_offset < screen_height) {
                        if j == 0 {
                            buffer[y_offset * screen_width + x] = blend_colors(
                                color,
                                buffer[y_offset * screen_width + x],
                                1.0 - dist,
                            );
                        } else if j == thickness {
                            buffer[y_offset * screen_width + x] =
                                blend_colors(color, buffer[y_offset * screen_width + x], dist);
                        } else {
                            buffer[y_offset * screen_width + x] = color;
                        }
                    }
                }
            }
            // Same as a bove but anti-aliasing logic apply on x instead of y
            // when line is vertical.
        } else {
            if pt2.1 < pt1.1 {
                (pt1, pt2) = (pt2, pt1);
            }
            let dx = pt2.0 - pt1.0;
            let dy = pt2.1 - pt1.1;
            let m = if dx != 0.0 {
                dx / dy // Compute the slope ratio.
            } else {
                dx / 1.0 // if dy == 0 then replace by 1.
            };
            for i in 0..=(dy.ceil() as usize) {
                let frac_x = pt1.0 + (i as f64) * m;
                let frac_y = pt1.1 + (i as f64);
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_x - (x as f64);
                ////////////////////////////////////////////////////////////////
                for j in 0..=thickness {
                    let x_offset = (x + j).saturating_sub(half_thickness) + 1;

                    if (x_offset < screen_width) && (y < screen_height) {
                        if j == 0 {
                            buffer[y * screen_width + x_offset] = blend_colors(
                                color,
                                buffer[y * screen_width + x_offset],
                                1.0 - dist,
                            );
                        } else if j == thickness {
                            buffer[y * screen_width + x_offset] =
                                blend_colors(color, buffer[y * screen_width + x_offset], dist);
                        } else {
                            buffer[y * screen_width + x_offset] = color;
                        }
                    }
                }
                ////////////////////////////////////////////////////////////////
            }
        }
    }

    /// Xiaolin Wu's line algorithm
    /// A good function for drawing clean
    /// anti-aliased line (without thickness though).
    pub fn draw_aa_line(
        buffer: &mut Vec<u32>,
        screen_width: usize,
        mut pt1: (f64, f64),
        mut pt2: (f64, f64),
        color: u32,
    ) {
        let screen_height = buffer.len() / screen_width;
        if (pt2.1 - pt1.1).abs() < (pt2.0 - pt1.0).abs() {
            // - Swap x and y for reversing the write order of the line.
            // if pt2(x,y) about pt1(x,y) is first from buffer logic
            // (buffer alway write from left to write) the process is writed
            // in reverse by swaping x an y.
            // so pt1 become pt2 and pt1 become pt2.
            if pt2.0 < pt1.0 {
                (pt1, pt2) = (pt2, pt1);
            }
            // Compute end point distances on x and y.
            let dx = pt2.0 - pt1.0;
            let dy = pt2.1 - pt1.1;
            ///////////////////////////////////////////
            // Avoid division by zero for
            // defining the slope ration m.
            let m = if dx != 0.0 {
                dy / dx // compute the slope ratio.
            } else {
                dy / 1.0 // if dx == 0 replace by 1.
            };
            //////////////////////////////////////////
            // Compute the x overlap distance for start point.
            let overlap = 1.0 - ((pt1.0 + 0.5) - ((pt1.0 + 0.5) as usize) as f64);
            // Vertical distance on y for the first point.
            let diststart = pt1.1 - ((pt1.1 as usize) as f64);
            // Write buffer only for the first point if input point are in the screen.
            let x = (pt1.0 + 0.5) as usize;
            let y = pt1.1 as usize;
            if (x < screen_width) && ((y + 1) < screen_height) {
                buffer[y * screen_width + x] = blend_colors(
                    color,
                    buffer[y * screen_width + x],
                    (1.0 - diststart) * overlap,
                );
                buffer[(y + 1) * screen_width + x] = blend_colors(
                    color,
                    buffer[(y + 1) * screen_width + x],
                    diststart * overlap,
                );
            }
            // Compute the x overlap distance for End point.
            let overlap = (pt2.0 - 0.5) - ((pt2.0 - 0.5) as usize) as f64;
            // Vertical distance on y for the first point.
            let distend = pt2.1 - ((pt2.1 as usize) as f64);
            // write buffer only for the first point if input point are in the screen.
            let x = (pt2.0 + 0.5) as usize;
            let y = pt2.1 as usize;
            if (x < screen_width) && ((y + 1) < screen_height) {
                buffer[y * screen_width + x] = blend_colors(
                    color,
                    buffer[y * screen_width + x],
                    (1.0 - distend) * overlap,
                );
                buffer[(y + 1) * screen_width + x] =
                    blend_colors(color, buffer[(y + 1) * screen_width + x], distend * overlap);
            }
            //////////////////////////////////
            // From 2nd point to penultimate point.
            for i in 1..=((dx) as usize) {
                // Move x px from + i on x axis
                let frac_x = pt1.0 + (i as f64);
                // Move the y px from slope ratio time
                // the n iteration step as scalar factor.
                let frac_y = pt1.1 + (i as f64) * m;
                // Convert x and y in integer.
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_y - (y as f64); // Get only the fractional part.
                if x < screen_width && (y + 1 < screen_height) {
                    // Apply opacity on alpha from fractional part distance as reminder of 1.
                    buffer[y * screen_width + x] =
                        blend_colors(color, buffer[y * screen_width + x], 1.0 - dist);
                    buffer[(y + 1) * screen_width + x] =
                        blend_colors(color, buffer[(y + 1) * screen_width + x], dist);
                }
            }
            // Same as a bove but anti-aliasing logic apply on x instead of y
            // when line is vertical.
        } else {
            if pt2.1 < pt1.1 {
                (pt1, pt2) = (pt2, pt1);
            }
            let dx = pt2.0 - pt1.0;
            let dy = pt2.1 - pt1.1;
            let m = if dx != 0.0 {
                dx / dy // compute the slope.
            } else {
                dx / 1.0 // slope if dx == 0.
            };
            // Compute the y overlap distance for start point.
            let overlap = 1.0 - ((pt1.1 + 0.5) - ((pt1.1 + 0.5) as usize) as f64);
            // Vertical distance on y for the first point.
            let diststart = pt1.1 - ((pt1.1 as usize) as f64);
            // write buffer only for the first point if input point are in the screen.
            let x = (pt1.0 + 0.5) as usize;
            let y = pt1.1 as usize;
            if (x < screen_width) && ((y + 1) < screen_height) {
                buffer[y * screen_width + x] = blend_colors(
                    color,
                    buffer[y * screen_width + x],
                    (1.0 - diststart) * overlap,
                );
                buffer[(y + 1) * screen_width + x] = blend_colors(
                    color,
                    buffer[(y + 1) * screen_width + x],
                    diststart * overlap,
                );
            }
            // Compute the y overlap distance for End point.
            let overlap = (pt2.1 - 0.5) - ((pt2.1 - 0.5) as usize) as f64;
            // Vertical distance on y for the first point.
            let distend = pt2.1 - ((pt2.1 as usize) as f64);
            // Write buffer only for the End point if input point are in the screen.
            let x = pt2.0 as usize;
            let y = (pt2.1 + 0.5) as usize;
            if ((x + 1) < screen_width) && (y < screen_height) {
                buffer[y * screen_width + x] = blend_colors(
                    color,
                    buffer[y * screen_width + x],
                    (1.0 - distend) * overlap,
                );
                buffer[y * screen_width + (x + 1)] =
                    blend_colors(color, buffer[y * screen_width + (x + 1)], distend * overlap);
            }
            for i in 1..=((dy) as usize) {
                let frac_x = pt1.0 + (i as f64) * m;
                let frac_y = pt1.1 + (i as f64);
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_x - (x as f64);
                if ((x + 1) < screen_width) && (y < screen_height) {
                    buffer[y * screen_width + x] =
                        blend_colors(color, buffer[y * screen_width + x], 1.0 - dist);
                    buffer[y * screen_width + (x + 1)] =
                        blend_colors(color, buffer[y * screen_width + (x + 1)], dist);
                }
            }
        }
    }

    /// Draw an anti aliased point designed for iteration writing from list
    /// 0.8 as aa_offset work well.
    pub fn draw_anti_aliased_point(
        buffer: &mut Vec<u32>,
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        aa_offset: f64,
        color: u32,
    ) {
        // Convert usize to f64 for fractional weight evaluation.
        let x_floor = x as f64;
        let y_floor = y as f64;
        let x = x as f64 + aa_offset;
        let y = y as f64 + aa_offset;

        // Define the fractional part.
        let x_frac = x - x_floor;
        let y_frac = y - y_floor;

        // Determine the surrounding pixel positions
        let x0 = x_floor as isize;
        let y0 = y_floor as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        // Define the weights for blending based on
        // the area covrage of the point to the pixel
        // centers (weight,offset x,offset y).
        let weights = [
            ((1.0 - x_frac) * (1.0 - y_frac), x0, y0), // Top-left pixel
            (x_frac * (1.0 - y_frac), x1, y0),         // Top-right pixel
            ((1.0 - x_frac) * y_frac, x0, y1),         // Bottom-left pixel
            (x_frac * y_frac, x1, y1),                 // Bottom-right pixel
        ];

        // Iterate over each surrounding pixel and apply the blended color
        for &(weight, xi, yi) in &weights {
            // Ensure the pixel is within the image boundaries
            if xi >= 0 && (xi as usize) < width && yi >= 0 && (yi as usize) < height {
                // Calculate the index in the buffer
                let index = (yi as usize * width + xi as usize) as usize;
                // Blend the current pixel color with the new color based on the calculated weight
                buffer[index] = blend_colors(color, buffer[index], weight);
            }
        }
    }

    /// Draw a very basic text for minimal feedback informations
    /// caution ! not all characters are implemented yet (see the list just bellow).
    /// but numerical values are.
    pub fn draw_text(
        buffer: &mut Vec<u32>,
        height: usize,
        width: usize,
        x: usize,
        y: usize,
        text: &str,
        scale: usize,
        text_color: u32,
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
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let sx = px + dx;
                                let sy = py + dy;
                                if sx < width && sy < height {
                                    buffer[sy * width + sx] = text_color; // White pixel
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /// Draw a unit Grid System with anti aliased lines for visual reference of the unit system.
    /// note:
    ///    - unid grid spacing should be always a multiple of the overall
    ///    both grid lengths related sides... ( a rounding process would
    ///    add a runtime overhead for each calls so it should be done outside
    ///    the method once and for all ( this method will alway be call from a
    ///    carefully designed runtime anyway so the grid specs have
    ///    to be computed form there ).
    ////////////////////////////////////////////////////////////////////////////
    pub fn draw_unit_grid_system(
        buffer: &mut Vec<u32>,
        screen_width: usize,
        screen_height: usize,
        background_color: u32,
        grid_global_alpha: f64,
        grid_plane: &CPlane,
        camera: &Camera,
        matrix: Option<&[[f64; 4]; 3]>,
        grid_x_length: f64,
        grid_y_length: f64,
        grid_spacing_unit: f64,
    ) {
        use super::intersection::clip_line;
        use super::transformation;
        ///////////////////////////////////////////////////////////
        let green_line_count = (grid_x_length / grid_spacing_unit) as usize + 1;
        let red_line_count = (grid_y_length / grid_spacing_unit) as usize + 1;
        let mut red_lines = Vec::new();
        let mut green_lines = Vec::new();
        ///////////////////////////////////////////////////////////
        // Put x and y line (Start,End) point in two stacks
        //         horizontal(red) and vertical(green)
        //    (since they can have different size length)
        ///////////////////////////////////////////////////////////
        // First stack:
        for i in 0..green_line_count {
            green_lines.push((
                grid_plane
                    .point_on_plane_uv(
                        (-(grid_x_length / 2.0)) + (grid_spacing_unit * (i as f64)),
                        -grid_y_length / 2.0,
                    )
                    .to_vertex(),
                grid_plane
                    .point_on_plane_uv(
                        (-(grid_x_length / 2.0)) + (grid_spacing_unit * (i as f64)),
                        grid_y_length / 2.0,
                    )
                    .to_vertex(),
            ));
        }
        // Second stack.
        for i in 0..red_line_count {
            red_lines.push((
                grid_plane
                    .point_on_plane_uv(
                        -grid_x_length / 2.0,
                        (-(grid_y_length / 2.0)) + (grid_spacing_unit * (i as f64)),
                    )
                    .to_vertex(),
                grid_plane
                    .point_on_plane_uv(
                        grid_x_length / 2.0,
                        (-(grid_y_length / 2.0)) + (grid_spacing_unit * (i as f64)),
                    )
                    .to_vertex(),
            ));
        }
        // Apply transformation matrix if needed.
        if let Some(matrix) = matrix {
            for i in 0..red_lines.len() {
                red_lines[i].0 = transformation::transform_point_4x3(matrix, &red_lines[i].0);
                red_lines[i].1 = transformation::transform_point_4x3(matrix, &red_lines[i].1);
            }
            for i in 0..green_lines.len() {
                green_lines[i].0 = transformation::transform_point_4x3(matrix, &green_lines[i].0);
                green_lines[i].1 = transformation::transform_point_4x3(matrix, &green_lines[i].1);
            }
        }
        // Project clipped lines on screen space.
        // For x aligned lignes (red).
        for line in red_lines.iter() {
            //////////////////////////////////////////////////////////////////////
            // Project line Start and End point on screen space.
            let line_point = (
                camera.project_maybe_outside(&line.0),
                camera.project_maybe_outside(&line.1),
            );
            if let Some(pt) = clip_line(line_point.0, line_point.1, screen_width, screen_height) {
                draw_aa_line(
                    buffer,
                    screen_width,
                    pt.0,
                    pt.1,
                    blend_colors(0x141414, background_color, 0.3 * grid_global_alpha),
                );
            }
        }
        // For y aligned lignes (green)
        for line in green_lines.iter() {
            //////////////////////////////////////////////////////////////////////
            // Project line Start and End point on screen space.
            let line_point = (
                camera.project_maybe_outside(&line.0),
                camera.project_maybe_outside(&line.1),
            );
            if let Some(pt) = clip_line(line_point.0, line_point.1, screen_width, screen_height) {
                draw_aa_line(
                    buffer,
                    screen_width,
                    pt.0,
                    pt.1,
                    blend_colors(0x141414, background_color, 0.3 * grid_global_alpha),
                );
            }
        }
        /*
         * - The following will draw (Red and Green) axis always
         *   from the midle of the grid if the lines count
         *   number is odd.
         */
        // Compute the grid line X axis.
        let mut u_points = [
            grid_plane.origin + (grid_plane.u * (grid_x_length * 0.5)),
            grid_plane.origin + (-(grid_plane.u) * (grid_x_length * 0.5)),
        ];
        // Compute the grid line Y axis.
        let mut v_points = [
            grid_plane.origin + (grid_plane.v * (grid_y_length * 0.5)),
            grid_plane.origin + (-(grid_plane.v) * (grid_y_length * 0.5)),
        ];
        // If there is a transformation matrix then transform the points.
        if let Some(matrix) = matrix {
            u_points[0] = transformation::transform_point_4x3(matrix, &u_points[0]);
            u_points[1] = transformation::transform_point_4x3(matrix, &u_points[1]);
            v_points[0] = transformation::transform_point_4x3(matrix, &v_points[0]);
            v_points[1] = transformation::transform_point_4x3(matrix, &v_points[1]);
        }
        //////////////////////////////////////////////////////////////////////
        // Project u axis line (from Start and End point)  (red line aligned and clipped on screen space).
        let mut line_point = (
            camera.project_maybe_outside(&u_points[0].to_vertex()),
            camera.project_maybe_outside(&u_points[1].to_vertex()),
        );
        if let Some(pt) = clip_line(line_point.0, line_point.1, screen_width, screen_height) {
            //draw_aa_line_with_thickness(buffer, screen_width, pt.0, pt.1, 2, 0x964b4b);
            draw_aa_line_with_thickness(
                buffer,
                screen_width,
                pt.0,
                pt.1,
                2,
                blend_colors(0x964b4b, background_color, grid_global_alpha),
            );
        }
        //////////////////////////////////////////////////////////////////////
        // Project v axis line (from Start and End point)  (green line aligned and clipped on screen space).
        line_point = (
            camera.project_maybe_outside(&v_points[0].to_vertex()),
            camera.project_maybe_outside(&v_points[1].to_vertex()),
        );
        if let Some(pt) = clip_line(line_point.0, line_point.1, screen_width, screen_height) {
            draw_aa_line_with_thickness(
                buffer,
                screen_width,
                pt.0,
                pt.1,
                2,
                blend_colors(0x4b964b, background_color, grid_global_alpha),
            );
        }
    }

    /// Draw a gimball. from a CPlane              //////////////////////
    /// Arrow are optional via boolean toggle (less runtime overhead.)
    /// this his only the graphical part of the incoming
    /// gimball object.
    /// TODO: make a gimbal object with some selectable parts and translation
    /// and rotation vector output when moved by user.
    /// also add circle for rotation handles.
    pub fn draw_gimball_from_plane(
        buffer: &mut Vec<u32>,
        screen_width: usize,
        screen_height: usize,
        background_color: u32,
        plane: &CPlane,
        camera: &Camera,
        matrix: Option<&[[f64; 4]; 3]>,
        scalar: f64,
        alpha: f64,
        draw_arrow: bool,
    ) {
        if alpha != 0.0 {
            use super::intersection::clip_line;
            use super::transformation;
            // Extract and scale cplane base components.
            let mut cplane_origin = plane.origin.to_vertex();
            let mut cplane_x_axis = (plane.origin + (plane.u * scalar)).to_vertex();
            let mut cplane_y_axis = (plane.origin + (plane.v * scalar)).to_vertex();
            let mut cplane_z_axis = (plane.origin + (plane.normal * scalar)).to_vertex();
            // Apply matrix transformation if needed.
            if let Some(matrix) = matrix {
                cplane_origin = transformation::transform_point_4x3(matrix, &cplane_origin);
                cplane_x_axis = transformation::transform_point_4x3(matrix, &cplane_x_axis);
                cplane_y_axis = transformation::transform_point_4x3(matrix, &cplane_y_axis);
                cplane_z_axis = transformation::transform_point_4x3(matrix, &cplane_z_axis);
            }
            // Project Cplane system on screen space.
            let cplane_origin_2dpoint = camera.project_maybe_outside(&cplane_origin);
            let cplane_x_axis_2dpoint = camera.project_maybe_outside(&cplane_x_axis);
            let cplane_y_axis_2dpoint = camera.project_maybe_outside(&cplane_y_axis);
            let cplane_z_axis_2dpoint = camera.project_maybe_outside(&cplane_z_axis);
            // Draw antialiased lines for each base axis colors.
            // TODO: make a refined layer aproch for alpha channel.
            if let Some(line_point) = clip_line(
                cplane_origin_2dpoint,
                cplane_x_axis_2dpoint,
                screen_width,
                screen_height,
            ) {
                draw_aa_line_with_thickness(
                    buffer,
                    screen_width,
                    line_point.0,
                    line_point.1,
                    2,
                    blend_colors(0xff0000, background_color, alpha),
                );
            }
            if let Some(line_point) = clip_line(
                cplane_origin_2dpoint,
                cplane_y_axis_2dpoint,
                screen_width,
                screen_height,
            ) {
                draw_aa_line_with_thickness(
                    buffer,
                    screen_width,
                    line_point.0,
                    line_point.1,
                    2,
                    blend_colors(0x00ff00, background_color, alpha),
                );
            }
            if let Some(line_point) = clip_line(
                cplane_origin_2dpoint,
                cplane_z_axis_2dpoint,
                screen_width,
                screen_height,
            ) {
                draw_aa_line_with_thickness(
                    buffer,
                    screen_width,
                    line_point.0,
                    line_point.1,
                    2,
                    blend_colors(0x0000ff, background_color, alpha),
                );
            }
            if draw_arrow {
                // Draw arrows by axis colors.
                let mut arrow_x = [
                    (Vertex::new(0.000, 0.000, -0.083) + Vertex::new(1.0, 0.0, 0.0)) * scalar,
                    (Vertex::new(0.000, -0.000, 0.083) + Vertex::new(1.0, 0.0, 0.0)) * scalar,
                    (Vertex::new(0.250, 0.000, -0.000) + Vertex::new(1.0, 0.0, 0.0)) * scalar,
                ];
                let mut arrow_y = [
                    (Vertex::new(-0.000, 0.000, -0.083) + Vertex::new(0.0, 1.0, 0.0)) * scalar,
                    (Vertex::new(0.000, 0.000, 0.083) + Vertex::new(0.0, 1.0, 0.0)) * scalar,
                    (Vertex::new(0.000, 0.250, -0.000) + Vertex::new(0.0, 1.0, 0.0)) * scalar,
                ];
                let mut arrow_z = [
                    (Vertex::new(0.083, 0.000, 0.000) + Vertex::new(0.0, 0.0, 1.0)) * scalar,
                    (Vertex::new(-0.083, 0.000, 0.000) + Vertex::new(0.0, 0.0, 1.0)) * scalar,
                    (Vertex::new(0.000, 0.000, 0.250) + Vertex::new(0.0, 0.0, 1.0)) * scalar,
                ];
                // Mutate arrow points arrays to map 3d points on the gimball CPlane local coordinates.
                arrow_x.iter_mut().for_each(|vertex| {
                    *vertex = plane
                        .point_on_plane(vertex.x, vertex.y, vertex.z)
                        .to_vertex()
                });
                arrow_y.iter_mut().for_each(|vertex| {
                    *vertex = plane
                        .point_on_plane(vertex.x, vertex.y, vertex.z)
                        .to_vertex()
                });
                arrow_z.iter_mut().for_each(|vertex| {
                    *vertex = plane
                        .point_on_plane(vertex.x, vertex.y, vertex.z)
                        .to_vertex()
                });
                // Apply transfomation matrix if needed.
                if let Some(matrix) = matrix {
                    arrow_x[0] = transformation::transform_point_4x3(matrix, &arrow_x[0]);
                    arrow_x[1] = transformation::transform_point_4x3(matrix, &arrow_x[1]);
                    arrow_x[2] = transformation::transform_point_4x3(matrix, &arrow_x[2]);
                    arrow_y[0] = transformation::transform_point_4x3(matrix, &arrow_y[0]);
                    arrow_y[1] = transformation::transform_point_4x3(matrix, &arrow_y[1]);
                    arrow_y[2] = transformation::transform_point_4x3(matrix, &arrow_y[2]);
                    arrow_z[0] = transformation::transform_point_4x3(matrix, &arrow_z[0]);
                    arrow_z[1] = transformation::transform_point_4x3(matrix, &arrow_z[1]);
                    arrow_z[2] = transformation::transform_point_4x3(matrix, &arrow_z[2]);
                }
                // Project 3d points from local CPlane coordinates to 2d
                // screen place.
                let arrow_x_2d = (
                    camera.project_maybe_outside(&arrow_x[0]),
                    camera.project_maybe_outside(&arrow_x[1]),
                    camera.project_maybe_outside(&arrow_x[2]),
                );
                let arrow_y_2d = (
                    camera.project_maybe_outside(&arrow_y[0]),
                    camera.project_maybe_outside(&arrow_y[1]),
                    camera.project_maybe_outside(&arrow_y[2]),
                );
                let arrow_z_2d = (
                    camera.project_maybe_outside(&arrow_z[0]),
                    camera.project_maybe_outside(&arrow_z[1]),
                    camera.project_maybe_outside(&arrow_z[2]),
                );
                // Clip line on 2d screen space for arrow x.
                if let Some(line_point) =
                    clip_line(arrow_x_2d.0, arrow_x_2d.1, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0xff0000, background_color, alpha),
                    );
                }
                if let Some(line_point) =
                    clip_line(arrow_x_2d.1, arrow_x_2d.2, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0xff0000, background_color, alpha),
                    );
                }
                if let Some(line_point) =
                    clip_line(arrow_x_2d.2, arrow_x_2d.0, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0xff0000, background_color, alpha),
                    );
                }
                // Clip line on 2d screen space for arrow y.
                if let Some(line_point) =
                    clip_line(arrow_y_2d.0, arrow_y_2d.1, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0x00ff00, background_color, alpha),
                    );
                }
                if let Some(line_point) =
                    clip_line(arrow_y_2d.1, arrow_y_2d.2, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0x00ff00, background_color, alpha),
                    );
                }
                if let Some(line_point) =
                    clip_line(arrow_y_2d.2, arrow_y_2d.0, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0x00ff00, background_color, alpha),
                    );
                }
                // Clip line on 2d screen space for arrow z.
                if let Some(line_point) =
                    clip_line(arrow_z_2d.0, arrow_z_2d.1, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0x0000ff, background_color, alpha),
                    );
                }
                if let Some(line_point) =
                    clip_line(arrow_z_2d.1, arrow_z_2d.2, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0x0000ff, background_color, alpha),
                    );
                }
                if let Some(line_point) =
                    clip_line(arrow_z_2d.2, arrow_z_2d.0, screen_width, screen_height)
                {
                    draw_aa_line_with_thickness(
                        buffer,
                        screen_width,
                        line_point.0,
                        line_point.1,
                        2,
                        blend_colors(0x0000ff, background_color, alpha),
                    );
                }
            }
        }
    }

    /// Make a contextual Grid of Vertex a unit system intervals.
    /// (the origin point is in the middle of the grid holding
    /// positive and negative domain on each relative sides.)
    /// - this produce an grid of point 3d focusing on unit sub sqare dimension
    /// rather than the sides overall length which may be not reached if sides length are
    /// not dividable by the unit spacing length.
    /// # Returns
    /// - an os memory allocated list of Vertex describing the UV grid in 3d space.
    pub fn make_3d_grid_from_center(
        plane: &CPlane,
        x_length: f64,
        y_length: f64,
        grid_spacing_unit: f64,
    ) -> Vec<Vertex> {
        let mut grid_points = Vec::new();
        let x_length = x_length / 2.0;
        let y_length = y_length / 2.0;
        let mut x = -x_length;
        let mut y = -y_length;
        while x <= x_length + std::f64::EPSILON {
            while y <= y_length + std::f64::EPSILON {
                grid_points.push((*plane).point_on_plane_uv(x, y).to_vertex());
                y += grid_spacing_unit;
            }
            if y >= y_length {
                y = -y_length;
            }
            x += grid_spacing_unit;
        }
        grid_points
    }

    /// Make a contextual Grid of Vertex from Construction Plane orientation.
    /// (the origin point is in the corner down left of the UV grid.)
    /// - the UV side dimension is divided equaly on u and v by a distance number
    /// unit system resulting a squared grid focussing on unit system sub square
    /// rather than the (maybe) non dividable overal sides length.
    /// # Returns
    /// - an os memory allocated list of Vertex describing the UV grid in 3d space.
    pub fn make_3d_grid_from_corner(
        plane: &CPlane,
        x_length: f64,
        y_length: f64,
        grid_spacing_unit: f64,
    ) -> Vec<Vertex> {
        let mut grid_points = Vec::new();
        let mut x = 0.0;
        let mut y = 0.0;
        while x <= (x_length + std::f64::EPSILON) {
            while y <= (y_length + std::f64::EPSILON) {
                grid_points.push((*plane).point_on_plane_uv(x, y).to_vertex());
                y += grid_spacing_unit;
            }
            x += grid_spacing_unit;
            if y >= y_length {
                y = 0.0;
            }
        }
        grid_points
    }

    /// Make a contextual Grid of Vertex from Construction Plane orientation.
    /// # Arguments
    /// - from CPlane
    /// - from sides UV dimensions length (f64) divided
    ///   by count integer numbers (usize) for U and V
    /// - this respect sides dimensions rather than unit grid cell.
    ///   the unit grid cell are divisions of each sides count numbers
    ///   respectivelly).
    /// -  the origin point is in the corner down left of the UV grid).
    /// # Returns
    /// - an os memory allocated list of Vertex describing the UV grid in 3d space
    ///   from construction plane.
    pub fn make_3d_divided_grid_from_corner(
        plane: &CPlane,
        u_length: f64,
        v_length: f64,
        divide_count_u: usize,
        divide_count_v: usize,
    ) -> Vec<Vertex> {
        // Evaluate from (u,v) dimension of the grid.
        let spacing_unit_u = u_length / (divide_count_u as f64);
        let spacing_unit_v = v_length / (divide_count_v as f64);

        // Define memory components.
        let mut grid_points = vec![Vertex::new(0.0, 0.0, 0.0); divide_count_u * divide_count_v];
        let mut pt_u = 0.0;
        let mut pt_v = 0.0;
        // Make a grid of points describing the plane.
        for v in 0..divide_count_v {
            for u in 0..divide_count_u {
                grid_points[v * divide_count_u + u] =
                    (*plane).point_on_plane_uv(pt_u, pt_v).to_vertex();
                pt_u += spacing_unit_u;
                if u == divide_count_u - 1 {
                    pt_u = 0.0;
                }
            }
            pt_v += spacing_unit_v;
        }
        grid_points
    }

    /// Draw a very basic rectangle very fast for screen space.
    pub fn draw_rectangle(
        buffer: &mut Vec<u32>,
        buffer_width: usize,
        buffer_height: usize,
        position_x: usize,
        position_y: usize,
        width: usize,
        height: usize,
        color: u32,
    ) {
        for y in position_y..position_y + height {
            for x in position_x..position_x + width {
                // Ensure rectangle is in the buffer space.
                if x < buffer_width && y < buffer_height {
                    buffer[y * buffer_width + x] = color;
                }
            }
        }
    }

    /// Draw a more advanced rectangle with antialiased rounded corner.
    pub fn draw_rounded_rectangle(
        buffer: &mut Vec<u32>,
        buffer_width: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        radius: usize,
        color: u32,
    ) {
        // Draw rounded corners with anti-aliasing
        draw_circle_quarter_aa(
            buffer,
            buffer_width,
            x + radius,
            y + radius,
            radius,
            color,
            true,
            true,
        ); // Top-left
        draw_circle_quarter_aa(
            buffer,
            buffer_width,
            x + width - radius - 1,
            y + radius,
            radius,
            color,
            false,
            true,
        ); // Top-right
        draw_circle_quarter_aa(
            buffer,
            buffer_width,
            x + radius,
            y + height - radius - 1,
            radius,
            color,
            true,
            false,
        ); // Bottom-left
        draw_circle_quarter_aa(
            buffer,
            buffer_width,
            x + width - radius - 1,
            y + height - radius - 1,
            radius,
            color,
            false,
            false,
        ); // Bottom-right
           // Fill horizontal parts (top and bottom)
        for dy in 0..radius {
            for dx in (x + radius)..(x + width - radius) {
                if x < buffer_width && y < buffer.len() / buffer_width {
                    buffer[(y + dy) * buffer_width + dx] = color;
                }
                if x < buffer_width && y < buffer.len() / buffer_width {
                    buffer[(y + height - dy - 1) * buffer_width + dx] = color;
                } // Bottom part
            }
        }
        // Fill vertical middle part
        for dy in (y + radius)..(y + height - radius) {
            for dx in x..(x + width) {
                //set_pixel(buffer, buffer_width, dx, dy, color);
                if x < buffer_width && y < buffer.len() / buffer_width {
                    buffer[dy * buffer_width + dx] = color;
                }
            }
        }
    }

    /// Draw a circle with anti-aliasing and minimal computing evaluations.
    /// # Arguments
    /// antialiasing_factor is a an offset value in pixel where the color is
    /// blended with the background color for smoother transition.  
    /// anti aliased factor add pixel to the bouduary circle
    /// note: if circle (radius + aa_offset) is out of the screen the
    /// app just not draw the circle.
    pub fn draw_circle(
        buffer: &mut Vec<u32>,
        screen_width: usize,
        screen_height: usize,
        center_x: usize,
        center_y: usize,
        radius: usize,
        thickness: usize,
        color: u32,
        antialiasing_factor: usize,
    ) {
        // Compute base components.
        ///////////////////////////////////////////////////////
        let aa_offset = if antialiasing_factor > 0 {
            1 * antialiasing_factor
        } else {
            1
        };
        // Define drawing boundary square area of the circle
        // to minimize computation.
        let radius_aa = radius + aa_offset;
        let inner_threshold = (radius - thickness - aa_offset) as f64;
        let fradius = radius as f64;
        let faa_offset = aa_offset as f64;
        // Define the boundary (x,y) limit of circle computation.
        let boundary_points_array = [
            (
                center_x.saturating_sub(radius_aa),
                center_y.saturating_sub(radius_aa),
            ),
            (center_x + radius_aa, center_y + radius_aa),
        ];
        ///////////////////////////////////////////////////////
        // Compute only square containing the circle.
        for y in (boundary_points_array[0].1)..=boundary_points_array[1].1 {
            for x in (boundary_points_array[0].0)..=(boundary_points_array[1].0) {
                // Compute Dx and Dy.
                let dx = (center_x as isize) - (x as isize);
                let dy = (center_y as isize) - (y as isize);
                let squared_ditance = ((dx * dx) + (dy * dy)) as f64;
                // Compute Alpha Value for outer range from radius.
                let alpha_out = 1.0
                    - utillity::remap(
                        (0.0, faa_offset),
                        (0.0, 1.0),
                        squared_ditance.sqrt() - fradius,
                    )
                    .clamp(0.0, 1.0);
                // Compute Alpha Value from inner range from radius.
                let alpha_in = utillity::remap(
                    (0.0, faa_offset),
                    (0.0, 1.0),
                    squared_ditance.sqrt() - inner_threshold,
                )
                .clamp(0.0, 1.0);
                // Evaluate circle border from alpha_in and alpha_out conditions.
                if (alpha_out > 0.0) && (alpha_in > 0.0) {
                    let alpha = if alpha_out < 1.0 {
                        alpha_out
                    } else if alpha_in < 1.0 {
                        alpha_in
                    } else {
                        alpha_in // (alpha is 1.0 in that case)
                    };
                    // Write the blended color to the buffer
                    if (x < screen_width) && (y < screen_height) {
                        let blended_color =
                            blend_colors(color, buffer[y * screen_width + x], alpha);
                        buffer[y * screen_width + x] = blended_color;
                    }
                }
            }
        }
    }

    /// Draw a full disc with anti aliasing for smooth visual effects.
    /// anti aliasing factor is a border transition added to the radius
    /// where pixel color is blended to the background color.
    /// note:
    /// if radius + aa factor
    /// (which represend the offset of the radius in pixel to compute transition)
    /// exceed the buffer frame coordinate screen resilotion the disc is just
    /// not rendered.
    pub fn draw_disc(
        buffer: &mut Vec<u32>,
        screen_width: usize,
        screen_height: usize,
        center_x: usize,
        center_y: usize,
        radius: usize,
        color: u32,
        antialiasing_factor: usize,
    ) {
        // Compute base components.
        ///////////////////////////////////////////////////////
        let aa_offset = if antialiasing_factor > 0 {
            1 * antialiasing_factor
        } else {
            1
        };
        // Define drawing boundary square area of the circle
        // to minimize computation.
        let radius_aa = radius + aa_offset;
        let fradius = radius as f64;
        let faa_offset = aa_offset as f64;
        // Define the boundary (x,y) limit of circle computation.
        let boundary_points_array = [
            (
                center_x.saturating_sub(radius_aa),
                center_y.saturating_sub(radius_aa),
            ),
            (center_x + radius_aa, center_y + radius_aa),
        ];
        ///////////////////////////////////////////////////////
        // Compute only square containing the circle.
        for y in (boundary_points_array[0].1)..=boundary_points_array[1].1 {
            for x in (boundary_points_array[0].0)..=(boundary_points_array[1].0) {
                // Compute Dx and Dy.
                let dx = (center_x as isize) - (x as isize);
                let dy = (center_y as isize) - (y as isize);
                let squared_ditance = ((dx * dx) + (dy * dy)) as f64;
                // Compute Alpha Value for outer range from radius.
                let alpha_out = 1.0
                    - utillity::remap(
                        (0.0, faa_offset),
                        (0.0, 1.0),
                        squared_ditance.sqrt() - fradius,
                    )
                    .clamp(0.0, 1.0);
                // Evaluate circle border from alpha_out value conditions.
                if alpha_out > 0.0 {
                    // Write the blended color to the buffer
                    if (x < screen_width) && (y < screen_height) {
                        let blended_color =
                            blend_colors(color, buffer[y * screen_width + x], alpha_out);
                        buffer[y * screen_width + x] = blended_color;
                    }
                }
            }
        }
    }

    // private Function to draw an anti-aliased quarter-circle for a specific corner
    fn draw_circle_quarter_aa(
        buffer: &mut Vec<u32>,
        buffer_width: usize,
        cx: usize,
        cy: usize,
        radius: usize,
        color: u32,
        is_left: bool,
        is_top: bool,
    ) {
        for y in 0..=radius {
            for x in 0..=radius {
                let distance = ((x * x + y * y) as f64).sqrt(); // Euclidean distance
                let dist_to_edge = radius as f64 - distance;
                if dist_to_edge >= 0.0 {
                    // Inside the circle
                    let px = if is_left { cx - x } else { cx + x };
                    let py = if is_top { cy - y } else { cy + y };
                    if x < buffer_width && y < buffer.len() / buffer_width {
                        buffer[py * buffer_width + px] = color;
                    }
                } else if dist_to_edge > -1.0 {
                    // Edge pixel: blend color
                    let alpha = 1.0 + dist_to_edge; // alpha ranges from 0 to 1
                    let px = if is_left { cx - x } else { cx + x };
                    let py = if is_top { cy - y } else { cy + y };
                    if x < buffer_width && y < buffer.len() / buffer_width {
                        buffer[py * buffer_width + px] =
                            blend_colors(color, buffer[y * buffer_width + x], alpha);
                    }
                }
            }
        }
    }

    #[inline(always)]
    /// Function to blend two colors based on alpha
    pub fn blend_colors(foreground: u32, background: u32, alpha: f64) -> u32 {
        let fg_r = ((foreground >> 16) & 0xFF) as f64;
        let fg_g = ((foreground >> 8) & 0xFF) as f64;
        let fg_b = (foreground & 0xFF) as f64;

        let bg_r = ((background >> 16) & 0xFF) as f64;
        let bg_g = ((background >> 8) & 0xFF) as f64;
        let bg_b = (background & 0xFF) as f64;

        let r = (alpha * fg_r + (1.0 - alpha) * bg_r) as u32;
        let g = (alpha * fg_g + (1.0 - alpha) * bg_g) as u32;
        let b = (alpha * fg_b + (1.0 - alpha) * bg_b) as u32;

        (0xFF << 24) | (r << 16) | (g << 8) | b // ARGB format
    }

    #[inline(always)]
    // after benchmark performance are revelated worst with the version of above.
    // compile time may be slightly faster though...
    pub fn blend_colors_deprecated(foreground: u32, background: u32, alpha: f64) -> u32 {
        // ARGB format
        (0xFF << 24)
            | (((alpha * (((foreground >> 16) & 0xFF) as f64)
                + (1.0 - alpha) * (((background >> 16) & 0xFF) as f64)) as u32)
                << 16)
            | (((alpha * (((foreground >> 8) & 0xFF) as f64)
                + (1.0 - alpha) * (((background >> 8) & 0xFF) as f64)) as u32)
                << 8)
            | ((alpha * ((foreground & 0xFF) as f64) + (1.0 - alpha) * ((background & 0xFF) as f64))
                as u32)
    }

    /// Describe a circle With Point 3d (this not draw the circle on buffer a function will be
    /// added soon for that.)
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
    pub struct Point2d {
        pub x: f64,
        pub y: f64,
    }

    impl Point2d {
        pub fn new(x: f64, y: f64) -> Self {
            Self { x, y }
        }
    }
    // below are private function for other methods
    // they will be api public whit better integration.
    fn draw_triangle_2d_v2(
        buffer: &mut Vec<u32>,
        width: usize,
        height: usize,
        p0: &Point2d,
        p1: &Point2d,
        p2: &Point2d,
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
        edge_start: &Point2d,
        edge_end: &Point2d,
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

    fn build_edges(p0: &Point2d, p1: &Point2d) -> Option<Edge> {
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

    /// Draw triangles from 3 vector2d points
    pub fn draw_triangle_optimized(
        buffer: &mut Vec<u32>,
        width: usize,
        height: usize,
        p0: &Point2d,
        p1: &Point2d,
        p2: &Point2d,
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
    //              following scratch codes deprecated.                   //
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    pub fn _exercise_draw_line_bcp(
        buffer: &mut Vec<u32>,
        screen_width: usize,
        mut pt1: (f64, f64),
        mut pt2: (f64, f64),
        color: u32,
    ) {
        if (pt2.1 - pt1.1).abs() < (pt2.0 - pt1.0).abs() {
            if pt2.0 < pt1.0 {
                (pt1, pt2) = (pt2, pt1);
            }
            let dx = pt2.0 - pt1.0;
            let dy = pt2.1 - pt1.1;
            /*
            // Compute overlap for the first point.
            let overlap = 1.0 - ((pt1.0+0.5) - (pt1.0 + 0.5).round());
            let diststart = pt1.1 - pt1.1.round();
            buffer[(pt1.1 as usize) * screen_width + ((pt1.0 + 0.5) as usize)] =
                    blend_colors(color,
                        buffer[(pt1.1 as usize) * screen_width + ((pt1.0 + 0.5) as usize)],
                        (1.0 - diststart) * overlap
                        );
            buffer[((pt1.1 as usize) + 1) * screen_width + ((pt1.0 + 0.5) as usize)] =
                    blend_colors(color,
                        buffer[(pt1.1 as usize) * screen_width + ((pt1.0 + 0.5) as usize)],
                         diststart * overlap
                        );

            // Compute overlap for the first point.
            let overlap = 1.0 - ((pt2.0-0.5) - (pt2.0 - 0.5).round());
            let distend = pt2.1 - pt2.1.round();
            buffer[((pt2.1 + 0.5) as usize) * screen_width + (pt1.0 as usize)] =
                    blend_colors(color,
                        buffer[((pt2.1 + 0.5) as usize) * screen_width + (pt1.0 as usize)],
                        (1.0 - distend) * overlap
                        );
            buffer[((pt2.1 as usize) + 1) * screen_width + ((pt2.0 + 0.5) as usize)] =
                    blend_colors(color,
                        buffer[(pt1.1 as usize) * screen_width + ((pt1.0 + 0.5) as usize)],
                         distend * overlap
                        );
            */
            //////////////////////////////////
            // avoid division by zero for
            // defining the slope ration m.
            let m = if dx != 0.0 {
                dy / dx // compute the slope.
            } else {
                dy / 1.0 // slope if dx == 0.
            };
            //////////////////////////////////
            for i in 0..(dx as usize) {
                let frac_x = pt1.0 + (i as f64);
                let frac_y = pt1.1 + (i as f64) * m;
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_y - (y as f64);
                buffer[y * screen_width + x] =
                    blend_colors(color, buffer[y * screen_width + x], 1.0 - dist);
                buffer[(y + 1) * screen_width + x] =
                    blend_colors(color, buffer[(y + 1) * screen_width + x], dist);
            }
        } else {
            if pt2.1 < pt1.1 {
                (pt1, pt2) = (pt2, pt1);
            }
            let dx = pt2.0 - pt1.0;
            let dy = pt2.1 - pt1.1;
            let m = if dy != 0.0 {
                dx / dy // compute the slope.
            } else {
                dx / 1.0 // slope if dx == 0.
            };
            for i in 0..(dy as usize) {
                let frac_x = pt1.0 + (i as f64) * m;
                let frac_y = pt1.1 + (i as f64);
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_x - (x as f64);
                buffer[y * screen_width + x] =
                    blend_colors(color, buffer[y * screen_width + x], 1.0 - dist);
                buffer[y * screen_width + x + 1] =
                    blend_colors(color, buffer[y * screen_width + x + 1], dist);
            }
        }
    }

    // A try to implement aa for this methods
    // not sure of the benefit.
    pub fn draw_thick_line_experimental(
        buffer: &mut Vec<u32>,
        width: usize,
        height: usize,
        start: (isize, isize),
        end: (isize, isize),
        color: u32,
        thickness: usize,
    ) {
        let (x0, y0) = start;
        let (x1, y1) = end;
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;
        let mut x = x0;
        let mut y = y0;
        let half_thickness = (thickness / 2) as isize;
        loop {
            // Draw the central line pixel
            if x >= 0 && x < width as isize && y >= 0 && y < height as isize {
                buffer[y as usize * width + x as usize] = color;
            }
            // Draw additional pixels to achieve the desired thickness
            for t in -half_thickness..=half_thickness {
                if dx > dy {
                    // Line is more horizontal; vary y-coordinate
                    if y + t >= 0 && y + t < height as isize {
                        draw_anti_aliased_point(
                            buffer,
                            width,
                            height,
                            x as usize,
                            (y + t) as usize,
                            0.6,
                            color,
                        );
                    }
                } else {
                    // Line is more vertical; vary x-coordinate
                    if x + t >= 0 && x + t < width as isize {
                        draw_anti_aliased_point(
                            buffer,
                            width,
                            height,
                            (x + t) as usize,
                            y as usize,
                            0.6,
                            color,
                        );
                    }
                }
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }

    pub fn draw_thick_line(
        buffer: &mut Vec<u32>,
        width: usize,
        height: usize,
        start: (isize, isize),
        end: (isize, isize),
        color: u32,
        thickness: usize,
    ) {
        let (x0, y0) = start;
        let (x1, y1) = end;
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;
        let mut x = x0;
        let mut y = y0;
        let half_thickness = (thickness / 2) as isize;
        loop {
            // Draw the central line pixel
            if x >= 0 && x < width as isize && y >= 0 && y < height as isize {
                buffer[y as usize * width + x as usize] = color;
            }
            // Draw additional pixels to achieve the desired thickness
            for t in -half_thickness..=half_thickness {
                if dx > dy {
                    // Line is more horizontal; vary y-coordinate
                    if y + t >= 0 && y + t < height as isize {
                        buffer[(y + t) as usize * width + x as usize] = color;
                    }
                } else {
                    // Line is more vertical; vary x-coordinate
                    if x + t >= 0 && x + t < width as isize {
                        buffer[y as usize * width + (x + t) as usize] = color;
                    }
                }
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Draws an anti-aliased line with a specified thickness on a pixel buffer
    /// using an adaptation of Xiaolin Wu's line algorithm.
    pub fn draw_anti_aliased_line(
        buffer: &mut Vec<u32>, // Mutable reference to the pixel buffer
        screen_width: usize,   // Width of the screen or image in pixels
        screen_height: usize,  // Height of the screen or image in pixels
        x0: usize,             // Starting x-coordinate of the line
        y0: usize,             // Starting y-coordinate of the line
        x1: usize,             // Ending x-coordinate of the line
        y1: usize,             // Ending y-coordinate of the line
        thickness: f64,        // Desired thickness of the line
        color: u32,            // Color of the line in ARGB format
    ) {
        // Closure to plot a pixel with a given intensity
        let mut plot = |x: usize, y: usize, intensity: f64| {
            // Ensure the pixel is within the screen boundaries
            if x < screen_width && y < screen_height {
                let idx = y * screen_width + x; // Calculate the buffer index
                let base_color = buffer[idx]; // Get the current pixel color
                let blended_color = blend_colors(color, base_color, intensity); // Blend colors
                buffer[idx] = blended_color; // Update the pixel color in the buffer
            }
        };

        // Convert integer coordinates to floating-point for precise calculations
        let (x0, y0, x1, y1) = (x0 as f64, y0 as f64, x1 as f64, y1 as f64);

        // Determine if the line is steep (i.e., slope > 1)
        let steep = (y1 - y0).abs() > (x1 - x0).abs();

        // If the line is steep, swap the x and y coordinates
        let (mut x0, mut y0, mut x1, mut y1) = if steep {
            (y0, x0, y1, x1)
        } else {
            (x0, y0, x1, y1)
        };

        // Ensure the line is drawn from left to right
        if x0 > x1 {
            (x0, x1) = (x1, x0);
            (y0, y1) = (y1, y0);
        }

        let dx = x1 - x0; // Difference in x-coordinates
        let dy = y1 - y0; // Difference in y-coordinates
        let gradient = if dx == 0.0 { 1.0 } else { dy / dx }; // Calculate the gradient

        // Handle the first endpoint
        let xend = x0.round(); // Round x0 to the nearest integer
        let yend = y0 + gradient * (xend - x0); // Calculate the corresponding y
        let xpxl1 = xend as usize; // Integer part of xend

        let mut intery = yend + gradient; // Initialize the y-intercept for the main loop

        // Draw the line with the specified thickness
        let half_thickness = thickness / 2.0;
        for x in xpxl1..=(x1.round() as usize) {
            // For each point on the centerline, draw a perpendicular "strip" of pixels
            for offset in -(half_thickness.ceil() as i32)..=(half_thickness.ceil() as i32) {
                let distance = (offset as f64).abs(); // Distance from the centerline
                let intensity = if distance <= half_thickness {
                    1.0 - (distance / half_thickness) // Linear falloff for intensity
                } else {
                    0.0
                };
                if steep {
                    // If the line is steep, plot transposed coordinates
                    plot(intery.floor() as usize + offset as usize, x, intensity);
                } else {
                    // Otherwise, plot normal coordinates
                    plot(x, intery.floor() as usize + offset as usize, intensity);
                }
            }
            intery += gradient; // Increment the y-intercept
        }
    }
}

pub mod coloring {
    #[derive(Debug, Clone, Copy)]
    pub struct Color {
        red: u8,
        green: u8,
        blue: u8,
        alpha: f32,
        value: Option<u32>,
        bg_color: Option<u32>,
        original_value: Option<u32>,
    }
    impl Color {
        // Constructor A (initially without caching).
        pub fn new_rgb_fast(red: u8, green: u8, blue: u8) -> Self {
            Self {
                red,
                green,
                blue,
                alpha: 1.0,
                value: None,
                bg_color: None,
                original_value: None,
            }
        }
        // Constructor B with cached absolute value.
        pub fn new_rgb(self, red: u8, green: u8, blue: u8) -> Self {
            let absolute_color = self.rgb_color(&red, &green, &blue);
            Self {
                red,
                green,
                blue,
                alpha: 1.0,
                value: Some(absolute_color),
                bg_color: None,
                original_value: None,
            }
        }
        // Constructor C (with alpha).
        pub fn new_rgb_a(
            self,
            red: u8,
            green: u8,
            blue: u8,
            mut alpha: f32,
            background_color: u32,
        ) -> Self {
            if alpha < 1.0 {
                let absolute_color =
                    self.rgba_color(&red, &green, &blue, &mut alpha, &background_color);
                Self {
                    red: ((absolute_color >> 16) & 0xFF) as u8,
                    green: ((absolute_color >> 8) & 0xFF) as u8,
                    blue: (absolute_color & 0xFF) as u8,
                    alpha,
                    value: Some(absolute_color),
                    bg_color: Some(background_color),
                    original_value: Some(
                        (red as u32) << 16 | ((green as u32) << 8) | (blue as u32),
                    ),
                }
            } else {
                Self {
                    red,
                    green,
                    blue,
                    alpha,
                    value: Some((red as u32) << 16 | ((green as u32) << 8) | (blue as u32)),
                    bg_color: None,
                    original_value: None,
                }
            }
        }

        /// Get absolute the absolute color value in u32.
        /// - compute & update alpha channel if nothing is cached..
        pub fn get_value(&mut self) -> u32 {
            if let Some(value) = self.value {
                value // return value if cached.
            } else {
                if self.alpha < 1.0 {
                    // Update and computed absolute color value from non opaque alpha
                    self.value = Some(self.rgba_color(
                        &self.red,
                        &self.green,
                        &self.blue,
                        &mut self.alpha,
                        &self.bg_color.unwrap(),
                    ));
                    // Backup original value.
                    self.original_value = Some(
                        (self.red as u32) << 16 | ((self.green as u32) << 8) | (self.blue as u32),
                    );
                    // Update RGB description components from updated absolute value.
                    self.red = ((self.value.unwrap() >> 16) & 0xFF) as u8;
                    self.green = ((self.value.unwrap() >> 8) & 0xFF) as u8;
                    self.blue = ((self.value.unwrap()) & 0xFF) as u8;
                    self.value.unwrap() // return the computed absolute value.
                } else {
                    // Update absolute value from RGB.
                    self.value = Some(
                        (self.red as u32) << 16 | ((self.green as u32) << 8) | (self.blue as u32),
                    );
                    self.value.unwrap() // Return absolute 32bit value.
                }
            }
        }

        /// Return Alpha Channel.
        pub fn get_alpha(self) -> f32 {
            self.alpha
        }

        /// Return Red component of value.
        pub fn get_red(self) -> u8 {
            self.red
        }

        /// Get Green.
        pub fn get_green(self) -> u8 {
            self.green
        }

        /// Get Blue.
        pub fn get_blue(self) -> u8 {
            self.blue
        }

        /// Mutate Color and Compute alpha from given background color.
        pub fn set_from_rgb_a_bg_components(
            &mut self,
            red: u8,
            green: u8,
            blue: u8,
            mut alpha: f32,
            bg_color: u32,
        ) {
            // Compute Alpha channel
            self.value = Some(self.rgba_color(&red, &green, &blue, &mut alpha, &bg_color));
            // Backup Original Value.
            self.original_value =
                Some((self.red as u32) >> 16 | (self.green as u32) >> 8 | (self.green as u32));
            // Update RGBA_BG.
            self.red = red;
            self.green = green;
            self.blue = blue;
            self.alpha = alpha;
            self.bg_color = Some(bg_color);
        }

        /// Remove alpha channel.
        pub fn set_opaque(&mut self) {
            if self.alpha < 1.0 {
                self.red = ((self.original_value.unwrap() >> 16) & 0xFF) as u8;
                self.green = ((self.original_value.unwrap() >> 8) & 0xFF) as u8;
                self.blue = ((self.original_value.unwrap()) & 0xFF) as u8;
                self.alpha = 1.0;
                self.original_value = None;
                self.value =
                    Some((self.red as u32) >> 16 | (self.green as u32) >> 8 | (self.green as u32));
            }
        }

        /// Mutate internals components and reset alpha state to opaque.
        pub fn set_from_absolute_value(&mut self, value: u32) {
            self.red = ((value >> 16) & 0xFF) as u8;
            self.green = ((value >> 8) & 0xFF) as u8;
            self.blue = (value & 0xFF) as u8;
            self.alpha = 1.0;
            self.original_value = None;
            self.value =
                Some((self.red as u32) >> 16 | (self.green as u32) >> 8 | (self.green as u32));
        }

        /// Convert an rgb value to minifb 0rgb standard.
        fn rgb_color(self, red: &u8, green: &u8, blue: &u8) -> u32 {
            (*red as u32) << 16 | (*green as u32) << 8 | (*blue as u32)
        }

        /// Blend rgb value with alpha to back ground color.
        /// # Arguments
        /// - red u8
        /// - green u8
        /// - blue u8
        /// - alpha f32 from 0.0 (transparent) to 1.0 (opaque)
        /// - Background_color u32 (color to blend with)
        /// # Returns
        /// return an RGB color blended with the background color with the alpha from 0.0 to 1.0.
        fn rgba_color(
            self,
            red: &u8,
            green: &u8,
            blue: &u8,
            alpha: &mut f32,
            bg_color: &u32,
        ) -> u32 {
            // Ensure alpha is clamped between 0.0 and 1.0
            (*alpha) = (*alpha).clamp(0.0, 1.0);

            // Extract background RGB components as f32.
            let bg_r = ((bg_color >> 16) & 0xFF) as f32;
            let bg_g = ((bg_color >> 8) & 0xFF) as f32;
            let bg_b = (bg_color & 0xFF) as f32;

            // Blend each channel
            let blended_r = ((*alpha) * (*red) as f32 + (1.0 - (*alpha)) * bg_r).round() as u32;
            let blended_g = ((*alpha) * (*green) as f32 + (1.0 - (*alpha)) * bg_g).round() as u32;
            let blended_b = ((*alpha) * (*blue) as f32 + (1.0 - (*alpha)) * bg_b).round() as u32;
            (blended_r << 16) | (blended_g << 8) | blended_b
        }

        /// Public and static version of the private method.  
        /// Convert R,G,B to an absolute value u32.
        pub fn convert_rgb_color(red: u8, green: u8, blue: u8) -> u32 {
            (red as u32) << 16 | (green as u32) << 8 | (blue as u32)
        }

        /// Get absolute color value from RGB, alpha and Background Color chanels.
        pub fn convert_rgba_color(
            red: u8,
            green: u8,
            blue: u8,
            mut alpha: f32,
            bg_color: u32,
        ) -> u32 {
            // Ensure alpha is clamped between 0.0 and 1.0
            alpha = alpha.clamp(0.0, 1.0);

            //TODO: a struct for Background color may be usefull
            // for caching in u8 form ....
            // Extract background RGB components as f32.
            let bg_r = ((bg_color >> 16) & 0xFF) as f32;
            let bg_g = ((bg_color >> 8) & 0xFF) as f32;
            let bg_b = (bg_color & 0xFF) as f32;

            // Blend each channel
            let blended_r = (alpha * red as f32 + (1.0 - alpha) * bg_r).round() as u32;
            let blended_g = (alpha * green as f32 + (1.0 - alpha) * bg_g).round() as u32;
            let blended_b = (alpha * blue as f32 + (1.0 - alpha) * bg_b).round() as u32;
            (blended_r << 16) | (blended_g << 8) | blended_b
        }

        /// Provide a new RGB value blended in Background color from Alpha
        /// vale 0.0 (Transparent) to 1.0 Opaque.
        pub fn convert_rgb_with_background_and_alpha(
            foreground: u32, // 0xRRGGBB
            background: u32, // 0xRRGGBB
            alpha: f32,      // Transparency: 0.0 (transparent) to 1.0 (opaque)
        ) -> u32 {
            // Ensure alpha is clamped between 0.0 and 1.0
            let alpha = alpha.clamp(0.0, 1.0);

            // Extract RGB components of the foreground
            let fg_r = ((foreground >> 16) & 0xFF) as f32;
            let fg_g = ((foreground >> 8) & 0xFF) as f32;
            let fg_b = (foreground & 0xFF) as f32;

            // Extract RGB components of the background
            let bg_r = ((background >> 16) & 0xFF) as f32;
            let bg_g = ((background >> 8) & 0xFF) as f32;
            let bg_b = (background & 0xFF) as f32;

            // Blend each channel
            let blended_r = (alpha * fg_r + (1.0 - alpha) * bg_r).round() as u32;
            let blended_g = (alpha * fg_g + (1.0 - alpha) * bg_g).round() as u32;
            let blended_b = (alpha * fg_b + (1.0 - alpha) * bg_b).round() as u32;

            // Recombine into a single u32 color
            (blended_r << 16) | (blended_g << 8) | blended_b
        }

        /// Convert a 32bit color to a 24 bit rounded u32 displayable color.
        /// this should avoid color banding artifact on 24 bits color display.
        pub fn find_closest24bits_color(color: u32) -> u32 {
            let r = ((((color >> 16) & 0xFF) as f32).round() as i32).clamp(0, 255) as u32;
            let g = ((((color >> 8) & 0xFF) as f32).round() as i32).clamp(0, 255) as u32;
            let b = (((color & 0xFF) as f32).round() as i32).clamp(0, 255) as u32;
            (r << 16) | (g << 8) | b
        }

        /// Convert to 24 bit space.
        pub fn convert_to_24_bit_space(color: u32) -> (u8, u8, u8) {
            let r = ((((color >> 16) & 0xFF) as f32).round() as i32).clamp(0, 255) as u8;
            let g = ((((color >> 8) & 0xFF) as f32).round() as i32).clamp(0, 255) as u8;
            let b = (((color & 0xFF) as f32).round() as i32).clamp(0, 255) as u8;
            (r, g, b)
        }

        pub fn buffer_filter_24bit_display_color(
            buffer: &mut Vec<u32>,
            screen_width: usize,
            screen_height: usize,
        ) {
            for y in 0..screen_height {
                for x in 0..screen_width {
                    buffer[y * screen_width + x] =
                        Self::find_closest24bits_color(buffer[y * screen_width + x]);
                }
            }
        }
    }
}
pub mod utillity {
    use core::f64;
    use std::ops::{Div, Sub};

    // Rust have already builtin function.
    pub fn degree_to_radians(input_angle_in_degre: &f64) -> f64 {
        (*input_angle_in_degre) * (f64::consts::PI * 2.0) / 360.0
    }
    pub fn radians_to_degree(input_angle_in_radians: &f64) -> f64 {
        (*input_angle_in_radians) * 360.0 / (f64::consts::PI * 2.0)
    }

    /// The famous Quake3 Arena algorithm.
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

    #[inline(always)]
    /// Round digit number to a precision scale.
    /// precision of 1_000.0 will round digit at 0.001
    /// precision of 1e6 will round at 0.000001
    /// the non fractional part is untouched.
    pub fn round_at_scale(input_value: f64, precision: f64) -> f64 {
        input_value.trunc() + (input_value.fract() * precision).round() / precision
    }

    /// Remap range 1 to range 2 from s value at the scale of range 1.
    pub fn remap(from_range: (f64, f64), to_range: (f64, f64), s: f64) -> f64 {
        to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
    }

    /// inverse linear interpolation... Normalize a range...
    /// used to find the relative position (between 0 and 1)
    /// of a value within a given range. This is useful for normalizing values.
    /// # Returns
    /// an Option<T> a normalized (t) parameters value from 0 to 1
    /// describing the interval from v_start to v_end.
    /// input is not clamped so the range will exceed interval linearly.
    /// T generic can be f64 usize i64 or what ever implementing Sub and Div and Copy traits.
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
    /// return the linear interpolation between two values
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
