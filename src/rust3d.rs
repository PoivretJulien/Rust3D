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
    use crate::render_tools::rendering_object::Vertex;
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

        /// Determines if a series of points lies on a straight line
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

        // the 3 following methods are alias for keeping
        // habits from other 3d Api.

        /// Same as unitize
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
                Vector3d::new(self.X / mag, self.Y / mag, self.Z / mag)
            } else {
                Vector3d::new(0.0, 0.0, 0.0)
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

    use std::fmt;
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
        pub fn rotate_point(&self, point: &Point3d) -> Point3d {
            let q_point = Quaternion::new(0.0, point.X, point.Y, point.Z);
            let q_conjugate = self.conjugate();
            let rotated_q = self.multiply(&q_point).multiply(&q_conjugate);

            Point3d::new(rotated_q.x, rotated_q.y, rotated_q.z)
        }

        pub fn rotate_point_around_axis(
            point: &Point3d,
            axis: &Point3d,
            angle_rad: f64,
        ) -> Point3d {
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

// Everything related to intersections or Zones evaluation.
pub mod intersection {
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
            let dx = (self.center_point.0 - test_point.0) as isize;
            let dy = (self.center_point.1 - test_point.1) as isize;
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

    use super::geometry::{CPlane, Point3d, Vector3d};
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
}

/*
 *  Most of the time for CAD applications rotating a CPlane
 *  is more than enough for operating geometry or CAD design.
 *  but for animation,simulation or robotic automation...
 *  where several transformations must occur one after the others,
 *  matrix transformation offer the possibility to be combined into only one
 *  and then being factorized over and array of points in a single shot
 *  reducing significantly the computation cost of the final transformation.
 */
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

/*
 * - Draw are all objects that are mainelly related to the 2d Screen space buffer
     like shapes, lines, gimball, world 3d grid ect they should be fast and
     are the main objects for user visual interactions.
   - this module is in developement and exploration... every thing may deeply
     change at  any time as i'm learning new technic for not only quality but efficiency
   - optiomization will come after an overall view of the module requirements.
*/
//TODO: the Draw method of a NurbsCurve must be in draw module.
pub mod draw {
    use super::geometry::Point3d;
    use crate::rust3d::utillity;
    use crate::{models_3d::FONT_5X7, render_tools::visualization_v3::coloring::Color};
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
    /// tickness line with thickness.
    /// will be rewrited with a better quality.
    pub fn draw_aa_line_with_thickness(
        buffer: &mut Vec<u32>,
        screen_width: usize,
        mut pt1: (f64, f64),
        mut pt2: (f64, f64),
        color: u32,
        thickness: usize,
    ) {
        let half_thickness = (thickness as f64) / 2.0;
        if (pt2.1 - pt1.1).abs() < (pt2.0 - pt1.0).abs() {
            if pt2.0 < pt1.0 {
                (pt1, pt2) = (pt2, pt1);
            }
            let dx = pt2.0 - pt1.0;
            let dy = pt2.1 - pt1.1;
            let m = if dx != 0.0 { dy / dx } else { dy / 1.0 };
            for i in 0..=(dx as usize) {
                let frac_x = pt1.0 + (i as f64);
                let frac_y = pt1.1 + (i as f64) * m;
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_y - (y as f64);

                for offset in -(half_thickness as isize)..=(half_thickness as isize) {
                    let y_offset = y as isize + offset;
                    if y_offset >= 0 && (y_offset as usize + 1) < buffer.len() / screen_width {
                        buffer[(y_offset as usize) * screen_width + x] = blend_colors(
                            color,
                            buffer[(y_offset as usize) * screen_width + x],
                            1.0 - dist,
                        );
                        buffer[(y_offset as usize + 1) * screen_width + x] = blend_colors(
                            color,
                            buffer[(y_offset as usize + 1) * screen_width + x],
                            dist,
                        );
                    }
                }
            }
        } else {
            if pt2.1 < pt1.1 {
                (pt1, pt2) = (pt2, pt1);
            }
            let dx = pt2.0 - pt1.0;
            let dy = pt2.1 - pt1.1;
            let m = if dy != 0.0 { dx / dy } else { dx / 1.0 };
            for i in 0..=(dy as usize) {
                let frac_x = pt1.0 + (i as f64) * m;
                let frac_y = pt1.1 + (i as f64);
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_x - (x as f64);
                for offset in -(half_thickness as isize)..=(half_thickness as isize) {
                    let x_offset = x as isize + offset;
                    if x_offset >= 0 && (x_offset as usize) < screen_width {
                        buffer[y * screen_width + (x_offset as usize)] = blend_colors(
                            color,
                            buffer[y * screen_width + (x_offset as usize)],
                            1.0 - dist,
                        );
                        buffer[y * screen_width + (x_offset as usize + 1)] = blend_colors(
                            color,
                            buffer[y * screen_width + (x_offset as usize) + 1],
                            dist,
                        );
                    }
                }
            }
        }
    }

    /// Xiaolin Wu's line algorithm
    /// A good function for drawing clean
    /// anti-aliased line without thickness.
    pub fn draw_aa_line(
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
                if x < screen_width && (y + 1 < (buffer.len() / screen_width)) {
                    buffer[y * screen_width + x] =
                        blend_colors(color, buffer[y * screen_width + x], 1.0 - dist);
                    buffer[(y + 1) * screen_width + x] =
                        blend_colors(color, buffer[(y + 1) * screen_width + x], dist);
                }
            }
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
            for i in 0..(dy as usize) {
                let frac_x = pt1.0 + (i as f64) * m;
                let frac_y = pt1.1 + (i as f64);
                let x = frac_x as usize;
                let y = frac_y as usize;
                let dist = frac_x - (x as f64);
                buffer[y * screen_width + x] =
                    blend_colors(color, buffer[y * screen_width + x], 1.0 - dist);
                buffer[(y + 1) * screen_width + x] =
                    blend_colors(color, buffer[(y + 1) * screen_width + x], dist);
            }
        }
    }

    /// Draw an antialiased point designed for iteration from list
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

    /// Draw a very basic text for bsic feedback infrormations
    /// caution ! not all characters are implemented yet (see the list just bellow).
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

    /// Draw a contextual Grid graphing a unit system intervals.
    /// (the origin point is in the middle of the grid holding
    /// positive and negative domain on each relative sides.)
    use super::geometry::CPlane;
    pub fn draw_3d_grid(
        plane: &CPlane,
        x_positif_length: f64,
        y_positif_length: f64,
        grid_spacing_unit: f64,
    ) -> Vec<Vertex> {
        let mut grid_points = Vec::new();
        let x_positif_length = x_positif_length / 2.0;
        let y_positif_length = y_positif_length / 2.0;
        let grid_spacing_unit = grid_spacing_unit / 2.0;
        let mut x = -x_positif_length;
        let mut y = -y_positif_length;
        while x <= x_positif_length {
            while y <= y_positif_length {
                grid_points.push((*plane).point_on_plane_uv(x, y).to_vertex());
                y += grid_spacing_unit;
            }
            if y >= y_positif_length {
                y = -y_positif_length;
            }
            x += grid_spacing_unit;
        }
        grid_points
    }

    /// Draw a very basic rectangle very fast.
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
            (center_x - radius_aa, center_y - radius_aa),
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
            (center_x - radius_aa, center_y - radius_aa),
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

    //private function to blend two colors based on alpha
    fn blend_colors(foreground: u32, background: u32, alpha: f64) -> u32 {
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

    use crate::render_tools::rendering_object::Vertex;
    use crate::render_tools::visualization_v3::Camera;

    /// Draw a Gimball from a CPlane and a scalar value.
    pub fn draw_plane_gimball_3d(
        mut buffer: &mut Vec<u32>,
        width: usize,
        plane: CPlane,
        camera: &Camera,
        alpha: f32,
        background_color: u32,
        scalar: f64,
    ) {
        // Check if the CPlane sytem is in the camera frame if yes draw bsis axis vectors of the system.
        if let Some(origin) = camera.project(plane.origin.to_vertex()) {
            if let Some(x_axis) = camera.project((plane.origin + (plane.u * scalar)).to_vertex()) {
                draw_line(
                    &mut buffer,
                    width,
                    (origin.0, origin.1),
                    (x_axis.0, x_axis.1),
                    Color::convert_rgba_color(255, 0, 0, alpha, background_color),
                );
            }
            if let Some(y_axis) = camera.project((plane.origin + (plane.v * scalar)).to_vertex()) {
                draw_line(
                    &mut buffer,
                    width,
                    (origin.0, origin.1),
                    (y_axis.0, y_axis.1),
                    Color::convert_rgba_color(0, 255, 0, alpha, background_color),
                );
            }
            if let Some(z_axis) =
                camera.project((plane.origin + (plane.normal * scalar)).to_vertex())
            {
                draw_line(
                    &mut buffer,
                    width,
                    (origin.0, origin.1),
                    (z_axis.0, z_axis.1),
                    Color::convert_rgba_color(0, 0, 255, alpha, background_color),
                );
            }
        }
        // Draw triangles arrows on CPlane (uvn)
        // below are unit triangles points from normalized basis system.
        let mut arrow_x = vec![
            (Vertex::new(0.000, 0.000, -0.083) + Vertex::new(1.0, 0.0, 0.0)) * scalar,
            (Vertex::new(0.000, -0.000, 0.083) + Vertex::new(1.0, 0.0, 0.0)) * scalar,
            (Vertex::new(0.250, 0.000, -0.000) + Vertex::new(1.0, 0.0, 0.0)) * scalar,
        ];
        // mutate the static data of triangles points on the CPlane location.
        arrow_x.iter_mut().for_each(|vertex| {
            *vertex = plane
                .point_on_plane(vertex.x, vertex.y, vertex.z)
                .to_vertex()
        });
        let mut arrow_y = vec![
            (Vertex::new(-0.000, 0.000, -0.083) + Vertex::new(0.0, 1.0, 0.0)) * scalar,
            (Vertex::new(0.000, 0.000, 0.083) + Vertex::new(0.0, 1.0, 0.0)) * scalar,
            (Vertex::new(0.000, 0.250, -0.000) + Vertex::new(0.0, 1.0, 0.0)) * scalar,
        ];
        arrow_y.iter_mut().for_each(|vertex| {
            *vertex = plane
                .point_on_plane(vertex.x, vertex.y, vertex.z)
                .to_vertex()
        });
        let mut arrow_z = vec![
            (Vertex::new(0.083, 0.000, 0.000) + Vertex::new(0.0, 0.0, 1.0)) * scalar,
            (Vertex::new(-0.083, 0.000, 0.000) + Vertex::new(0.0, 0.0, 1.0)) * scalar,
            (Vertex::new(0.000, 0.000, 0.250) + Vertex::new(0.0, 0.0, 1.0)) * scalar,
        ];
        arrow_z.iter_mut().for_each(|vertex| {
            *vertex = plane
                .point_on_plane(vertex.x, vertex.y, vertex.z)
                .to_vertex()
        });
        // Project arrows 3d system on 2d screen space.
        // if arrows are in screen space then draw the triangle with simple 2d lines.
        ///////////////////////////////////////////////////////////////////////////////////////
        let mut arrow_x_pt: Vec<(usize, usize)> = Vec::new();
        for i in 0..3usize {
            if let Some(pt) = camera.project(arrow_x[i]) {
                arrow_x_pt.push((pt.0, pt.1));
            }
        }
        if arrow_x_pt.len() == 3 {
            draw_line(
                &mut buffer,
                width,
                (arrow_x_pt[0].0, arrow_x_pt[0].1),
                (arrow_x_pt[1].0, arrow_x_pt[1].1),
                Color::convert_rgba_color(255, 0, 0, alpha, background_color),
            );
            draw_line(
                &mut buffer,
                width,
                (arrow_x_pt[0].0, arrow_x_pt[0].1),
                (arrow_x_pt[2].0, arrow_x_pt[2].1),
                Color::convert_rgba_color(255, 0, 0, alpha, background_color),
            );
            draw_line(
                &mut buffer,
                width,
                (arrow_x_pt[1].0, arrow_x_pt[1].1),
                (arrow_x_pt[2].0, arrow_x_pt[2].1),
                Color::convert_rgba_color(255, 0, 0, alpha, background_color),
            );
        }
        let mut arrow_y_pt: Vec<(usize, usize)> = Vec::new();
        for i in 0..3usize {
            if let Some(pt) = camera.project(arrow_y[i]) {
                arrow_y_pt.push((pt.0, pt.1));
            }
        }
        if arrow_y_pt.len() == 3 {
            draw_line(
                &mut buffer,
                width,
                (arrow_y_pt[0].0, arrow_y_pt[0].1),
                (arrow_y_pt[1].0, arrow_y_pt[1].1),
                Color::convert_rgba_color(0, 255, 0, alpha, background_color),
            );
            draw_line(
                &mut buffer,
                width,
                (arrow_y_pt[0].0, arrow_y_pt[0].1),
                (arrow_y_pt[2].0, arrow_y_pt[2].1),
                Color::convert_rgba_color(0, 255, 0, alpha, background_color),
            );
            draw_line(
                &mut buffer,
                width,
                (arrow_y_pt[1].0, arrow_y_pt[1].1),
                (arrow_y_pt[2].0, arrow_y_pt[2].1),
                Color::convert_rgba_color(0, 255, 0, alpha, background_color),
            );
        }
        let mut arrow_z_pt: Vec<(usize, usize)> = Vec::new();
        for i in 0..3usize {
            if let Some(pt) = camera.project(arrow_z[i]) {
                arrow_z_pt.push((pt.0, pt.1));
            }
        }
        if arrow_z_pt.len() == 3 {
            draw_line(
                &mut buffer,
                width,
                (arrow_z_pt[0].0, arrow_z_pt[0].1),
                (arrow_z_pt[1].0, arrow_z_pt[1].1),
                Color::convert_rgba_color(0, 0, 255, alpha, background_color),
            );
            draw_line(
                &mut buffer,
                width,
                (arrow_z_pt[0].0, arrow_z_pt[0].1),
                (arrow_z_pt[2].0, arrow_z_pt[2].1),
                Color::convert_rgba_color(0, 0, 255, alpha, background_color),
            );
            draw_line(
                &mut buffer,
                width,
                (arrow_z_pt[1].0, arrow_z_pt[1].1),
                (arrow_z_pt[2].0, arrow_z_pt[2].1),
                Color::convert_rgba_color(0, 0, 255, alpha, background_color),
            );
        }
        ///////////////////////////////////////////////////////////////////////////////////////
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

pub mod utillity {
    use core::f64;
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
 *   Unit Test for main structural components as far as i can.
 *   Every thing graphical is cumbersome to evalutate.
 *   but essential components will always be there since
 *   they are always crafted with love.
 */
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
    fn test_vector3d_are_perpendicular() {
        let vec_a = Vector3d::new(1.3, 1.55, 2.4);
        let vec_b = Vector3d::new(0.9, 1.25, 1.11);
        let vec_c = Vector3d::cross_product(&vec_a, &vec_b).unitize_b();
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
    fn project_vector_on_cplane() {
        let origin = Point3d::new(9.35028, 4.160783, -12.513656);
        let normal = Vector3d::new(-0.607828, -0.292475, -0.738244);
        let plane = CPlane::new(&origin, &normal);
        let vector = Vector3d::new(-1.883283, 2.49779, -6.130442);
        let expected_result = Point3d::new(10.469624, 8.103378, -14.997222);
        ////////////////////////////////////////////////////////////////////////
        // assert_eq!(expected_result,origin + vector.project_on_cplane(&plane));
        let result = origin + vector.project_on_cplane(&plane);
        if (result - expected_result).Length() <= 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
        ////////////////////////////////////////////////////////////////////////
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
        let mut plane_normal = Vector3d::new(0.65694, -0.31293, 0.685934);
        let plane = CPlane::new(&plane_origin, &plane_normal);
        let expected_result = Point3d::new(-9.583205, 5.873738, 10.838721);
        // assert_eq!(expected_result,intersect_line_with_plane(&point, &direction, &plane).unwrap());
        if let Some(result_point) = intersect_ray_with_plane(&point, &direction, &plane) {
            if (result_point - expected_result).Length().abs() < 1e-5 {
                assert!(true);
            } else {
                assert!(false);
                assert_eq!(result_point, expected_result);
            }
        } else {
            assert!(false);
        }
    }
    #[test]
    fn test_ray_trace_v_b() {
        use super::intersection::*;
        let point = Point3d::new(15.417647, 4.098069, 11.565836);
        let direction = Vector3d::new(-6.509447, -2.89155, -3.065556);
        let plane_origin = Point3d::new(-5.598372, -15.314516, -6.014116);
        let mut plane_normal = Vector3d::new(0.887628, 0.298853, 0.350434); //plane_normal = -plane_normal;
        let plane = CPlane::new(&plane_origin, &plane_normal);
        let expected_result = Point3d::new(-10.410072, -7.374817, -0.597459);
        // assert_eq!(expected_result,intersect_line_with_plane(&point, &direction, &plane).unwrap());
        if let Some(result_point) = intersect_ray_with_plane(&point, &direction, &plane) {
            if (result_point - expected_result).Length().abs() < 1e-4 {
                assert!(true);
            } else {
                assert_eq!(result_point, expected_result);
            }
        } else {
            assert!(false);
        }
    }
    use crate::render_tools::visualization_v3::coloring::Color;
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
            Triangle::new(&vertices, [0, 1, 2]),
            Triangle::new(&vertices, [0, 2, 3]),
        ];
        let mesh = Mesh::new_with_data(vertices, triangles);
        mesh.export_to_obj_with_normals_fast("./geometry/exported_light_with_rust.obj")
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

    use crate::render_tools::rendering_object::{Mesh, Triangle, Vertex};
    #[test]
    fn test_triangle_area() {
        // The following Triangle is flat in XY plane.
        let v1 = Vertex::new(1.834429, 0.0, -0.001996);
        let v2 = Vertex::new(1.975597, 0.0, 0.893012);
        let v3 = Vertex::new(2.579798, 0.0, 0.150466);
        let vertices = vec![v1, v2, v3];
        let tri = Triangle::new(&vertices, [0, 1, 2]);
        let expected_reuslt_area = 0.322794;
        let result = tri.get_triangle_area(&vertices);
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
    #[test]
    fn test_points_are_colinear() {
        // point 8 to 17 are a line (zero based).
        let p_tarray = vec![
            Point3d::new(1.575, 2.077, 1.777),
            Point3d::new(1.672, 2.240, 1.732),
            Point3d::new(1.765, 2.398, 1.663),
            Point3d::new(1.854, 2.549, 1.576),
            Point3d::new(1.940, 2.692, 1.474),
            Point3d::new(2.023, 2.828, 1.360),
            Point3d::new(2.104, 2.956, 1.236),
            Point3d::new(2.184, 3.077, 1.105),
            Point3d::new(2.276, 3.166, 0.961),
            Point3d::new(2.367, 3.255, 0.818),
            Point3d::new(2.459, 3.344, 0.675),
            Point3d::new(2.551, 3.433, 0.531),
            Point3d::new(2.643, 3.522, 0.388),
            Point3d::new(2.734, 3.611, 0.245),
            Point3d::new(2.826, 3.700, 0.102),
            Point3d::new(2.918, 3.789, -0.042),
            Point3d::new(3.010, 3.878, -0.185),
            Point3d::new(3.101, 3.967, -0.328),
            Point3d::new(3.193, 4.056, -0.472),
            Point3d::new(3.321, 4.120, -0.608),
            Point3d::new(3.459, 4.179, -0.738),
            Point3d::new(3.607, 4.232, -0.857),
            Point3d::new(3.767, 4.280, -0.964),
            Point3d::new(3.938, 4.321, -1.056),
            Point3d::new(4.118, 4.355, -1.128),
            Point3d::new(4.307, 4.383, -1.180),
            Point3d::new(4.502, 4.405, -1.208),
            Point3d::new(4.699, 4.422, -1.213),
            Point3d::new(4.896, 4.435, -1.194),
            Point3d::new(5.089, 4.446, -1.154),
            Point3d::new(5.277, 4.457, -1.094),
            Point3d::new(5.459, 4.468, -1.015),
            Point3d::new(5.632, 4.483, -0.921),
            Point3d::new(5.796, 4.501, -0.812),
            Point3d::new(5.951, 4.524, -0.691),
            Point3d::new(6.095, 4.553, -0.559),
            Point3d::new(6.228, 4.589, -0.417),
            Point3d::new(6.350, 4.633, -0.266),
            Point3d::new(6.458, 4.684, -0.109),
            Point3d::new(6.554, 4.743, 0.054),
        ];
        if Point3d::are_points_collinear(&p_tarray[8..19], 1e-3) {
            assert!(true);
        } else {
            assert!(false);
        }
        if !Point3d::are_points_collinear(&p_tarray[16..35], 1e-3) {
            assert!(true);
        } else {
            assert!(false);
        }
        // find the 3 first segments where the points array describe a straight line.
        if let Some(result) = Point3d::find_first_collinear_points(&p_tarray[0..], 1e-3) {
            assert_eq!((7, 10), result);
        }
    }
    use crate::render_tools::rendering_object::*;
    #[test]
    fn test_ray_trace() {
        let obj = Mesh::import_obj_with_normals("./geometry/flatbox.obj")
            .ok()
            .unwrap();
        let v_inside = Vertex::new(-0.130, -0.188, 2.327);
        assert!(v_inside.is_inside_a_mesh(&obj));
        let v_outside = Vertex::new(0.623, -0.587, 2.327);
        assert!(!v_outside.is_inside_a_mesh(&obj));

        let pt_origin = Vertex::new(1.240, -0.860, 3.169);
        let pt_direction = Vertex::new(-0.743, 0.414, -0.526);

        let ray = Ray::new(pt_origin, pt_direction);
        let bvh = BVHNode::build(&obj, (0..obj.triangles.len()).collect(), 0);
        for tri in obj.triangles.iter() {
            if let Some(t) = tri.intersect(&ray, &obj.vertices) {
                println!("---------->{t}");
            }
        }
        // Perform intersection test on bvh.
        if let Some((t, _vert)) = bvh.intersect(&obj, &ray) {
            //  _triangle is the ref to intersected triangle geometry.
            // *_triangle.intersect(&ray) (for refinements...)
            println!("Hit triangle at t = {}!", t);
            assert_eq!("1.436", format!("{t:0.3}").as_str());
        } else {
            println!("No intersection. {:?} {:?}", obj.vertices, ray);
            assert!(false);
        }
        assert!(true);
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
*
*  this describe mathematically the rotation of an unit vector on a orthogonal basis sytem.
*  - core mechanisum cos(theta) and sin(theta) will serve to divide unit basis axis length
*  by multiply the basix length of reference (x or y) by a number from 0.0 to 1.0 giving
*  a division of (x,y) component of the rotated vector.
*
*  -   we can also simplify this concept to the fact that a dot product of a unit vector by
*      the cos(theta) or sin(theta) produce it's projection on the base vector system
*      when vector start to rotate one part of the projection length for x ar y axis or
*      (their remider) is added or substracted on the oposit vector component
*      based on which axis we are operating on this discribe the rotation of the vector.
*
*  This produce: the 4x4 matrix rotation. where you distribute a Vector3d
*  disposed in colown to each menber of the row matrix  and add hem up
*  to produce the new rotated Vector3d.
*
*  the rotation identity:
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

   if someone one day read my study about transformation matrix:

   - it's dificult to warp your mind around without a step by step
     process but it's worth it to understand how this logic work
     at this end it's not so complex it's just that the step by step
     process lie throuhg a non negligable opaque abstraction if you don't
     involve yourself on the basis for a moment.
   - many are passioned by this., if you don't i guess you are on wrong place.
     so you may leave that topic with in mind that's the base of what at
     my sens we can call computing... either with a calculator a computer...
     so if you are programer it's just the leverage that you may give at your tools
     to amplify your computing capability.
     it's at the core of pation for "computing computers" it up to you
     to cheat with that or not.
*
*/
