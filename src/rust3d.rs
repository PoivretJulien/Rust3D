// *************************************************************************
// ******   First scratch of my basic lib for my computational need  *******
// *************************************************************************
#[allow(dead_code)]
pub mod geometry {
    // Implementation of a Point3d structure
    // bound to Vector3d structure
    // for standard operator processing.
    use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};
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
                Length: Vector3d::compute_length_byref(&x, &y, &z),
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
            Vector3d::new(
                self.X / self.Length,
                self.Y / self.Length,
                self.Z / self.Length,
            )
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
            if (*vector_a) * (*vector_b) == 0.0 {
                true
            } else {
                false
            }
        }

        /// Rotate a vector around an axis using Rodrigues' rotation formula.
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

        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        pub fn point_on_plane_uv(&self, u: &f64, v: &f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * (*u) + self.v.X * (*v),
                Y: self.origin.Y + self.u.Y * (*u) + self.v.Y * (*v),
                Z: self.origin.Z + self.u.Z * (*u) + self.v.Z * (*v),
            }
        }

        /// Converts local (u, v) coordinates to global (x, y, z) coordinates on the plane
        /// Also offsets the point along the plane's normal by z value.
        pub fn point_on_plane(&self, x: &f64, y: &f64, z: &f64) -> Point3d {
            Point3d {
                X: self.origin.X + self.u.X * (*x) + self.v.X * (*y) + self.normal.X * (*z),
                Y: self.origin.Y + self.u.Y * (*x) + self.v.Y * (*y) + self.normal.Y * (*z),
                Z: self.origin.Z + self.u.Z * (*x) + self.v.Z * (*y) + self.normal.Z * (*z),
            }
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
    /// (work in progress not tested yet)
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
pub mod visualization {

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

        // Multiply 3d point by a matrix vector.
        fn multiply_matrix_vector(&self, matrix: [[f64; 4]; 4], v: Point3d) -> Point3d {
            Point3d::new(
                matrix[0][0] * v.X + matrix[0][1] * v.Y + matrix[0][2] * v.Z + matrix[0][3],
                matrix[1][0] * v.X + matrix[1][1] * v.Y + matrix[1][2] * v.Z + matrix[1][3],
                matrix[2][0] * v.X + matrix[2][1] * v.Y + matrix[2][2] * v.Z + matrix[2][3],
            )
        }
    }

    /*
     *  Ray tracing algorithm study.
     *  Camera and vertex are for now overlapping objects
     *  they will be implemented in the rest of the code
     *  after some api and performances optimizations.
     *  (caching,references)
     *  for now a good sequencing process is the goal.
     */
    pub mod redering_object {

        // Todo: (optimize data structure)
        // Vertex is use as multi purpose usage
        // can be use as Point3d or Vector3d.
        #[derive(Debug, Copy, Clone)]
        pub struct Vertex {
            pub x: f64,
            pub y: f64,
            pub z: f64,
        }
        use std::hash::{Hash, Hasher};

        impl PartialEq for Vertex {
            fn eq(&self, other: &Self) -> bool {
                self.x == other.x && self.y == other.y && self.z == other.z
            }
        }

        impl Eq for Vertex {}

        impl Hash for Vertex {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.x.to_bits().hash(state);
                self.y.to_bits().hash(state);
                self.z.to_bits().hash(state);
            }
        }

        impl Vertex {
            pub fn new(x: f64, y: f64, z: f64) -> Self {
                Self { x, y, z }
            }

            // Add two vertices
            pub fn add(&self, other: &Vertex) -> Vertex {
                Vertex {
                    x: self.x + other.x,
                    y: self.y + other.y,
                    z: self.z + other.z,
                }
            }

            // Multiply a vertex by a scalar
            pub fn mul(&self, scalar: f64) -> Vertex {
                Vertex {
                    x: self.x * scalar,
                    y: self.y * scalar,
                    z: self.z * scalar,
                }
            }

            // Divide a vertex by a scalar
            pub fn div(&self, scalar: f64) -> Vertex {
                Vertex {
                    x: self.x / scalar,
                    y: self.y / scalar,
                    z: self.z / scalar,
                }
            }

            // Access the coordinate by axis index (0 = x, 1 = y, 2 = z)
            pub fn get_by_axis(&self, axis: usize) -> f64 {
                match axis {
                    0 => self.x,
                    1 => self.y,
                    2 => self.z,
                    _ => panic!("Invalid axis index!"), // You can handle this more gracefully if needed
                }
            }
            // Compute the magnitude (length) of the vector
            pub fn magnitude(&self) -> f64 {
                (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
            }

            // Normalize the vector (make it unit-length)
            pub fn unitize(&self) -> Vertex {
                let magnitude = self.magnitude();
                if magnitude == 0.0 {
                    panic!("Cannot normalize a zero-length vector");
                }
                Vertex {
                    x: self.x / magnitude,
                    y: self.y / magnitude,
                    z: self.z / magnitude,
                }
            }
        }

        #[derive(Debug, Copy, Clone)]
        pub struct Triangle {
            pub v0: Vertex,
            pub v1: Vertex,
            pub v2: Vertex,
            pub normal: Vertex, // Precomputed normal vector
        }

        impl Triangle {
            pub fn new(v0: Vertex, v1: Vertex, v2: Vertex) -> Self {
                // Represent edge 1 by a vector.
                let edge1 = Vertex {
                    x: v1.x - v0.x,
                    y: v1.y - v0.y,
                    z: v1.z - v0.z,
                };
                // Make Vector of edge 2.
                let edge2 = Vertex {
                    x: v2.x - v0.x,
                    y: v2.y - v0.y,
                    z: v2.z - v0.z,
                };
                // Normal is simply the cross Product of edge 1 and 2.
                let normal = Vertex::new(
                    edge1.y * edge2.z - edge1.z * edge2.y,
                    edge1.z * edge2.x - edge1.x * edge2.z,
                    edge1.x * edge2.y - edge1.y * edge2.x,
                );
                //.unitize();
                Self { v0, v1, v2, normal }
            }

            // Constructor with precomputed normal
            pub fn with_normal(v0: Vertex, v1: Vertex, v2: Vertex, normal: Vertex) -> Self {
                Self { v0, v1, v2, normal }
            }
        }

        use rayon::prelude::*;
        use std::collections::HashMap;
        use std::fs::File;
        use std::io::{self, BufRead, BufReader, BufWriter, Write};
        use std::iter::Iterator;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        use tobj; // For parallel iterators;

        #[derive(Debug)]
        pub struct Mesh {
            pub vertices: Vec<Vertex>,
            pub triangles: Vec<Triangle>,
        }

        impl Mesh {
            pub fn new(vertices: Vec<Vertex>, triangles: Vec<Triangle>) -> Self {
                Self {
                    vertices,
                    triangles,
                }
            }

            /// Display Statistics of obj file.
            /// # Arguments
            ///     file path of the obj file.
            /// # Returns
            ///     Result(vertex count,normal count,face count)
            pub fn count_obj_elements(file_path: &str) -> io::Result<(usize, usize, usize)> {
                let file = File::open(file_path)?;
                let reader = BufReader::new(file);

                let mut vertex_count = 0;
                let mut normal_count = 0;
                let mut face_count = 0;

                for line in reader.lines() {
                    let line = line?;
                    let mut words = line.split_whitespace();

                    if let Some(prefix) = words.next() {
                        match prefix {
                            "v" => vertex_count += 1,  // Vertex position
                            "vn" => normal_count += 1, // Vertex normal
                            "f" => face_count += 1,    // Face
                            _ => {}                    // Ignore other lines
                        }
                    }
                }

                Ok((vertex_count, normal_count, face_count))
            }

            /// Export the mesh to an .obj file.
            pub fn export_to_obj(&self, file_path: &str) -> io::Result<()> {
                let mut file = File::create(file_path)?;

                // Write vertices
                for vertex in &self.vertices {
                    writeln!(file, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
                }

                // Write faces
                for triangle in &self.triangles {
                    let v0_idx = self
                        .vertices
                        .iter()
                        .position(|v| *v == triangle.v0)
                        .unwrap()
                        + 1;
                    let v1_idx = self
                        .vertices
                        .iter()
                        .position(|v| *v == triangle.v1)
                        .unwrap()
                        + 1;
                    let v2_idx = self
                        .vertices
                        .iter()
                        .position(|v| *v == triangle.v2)
                        .unwrap()
                        + 1;

                    writeln!(file, "f {} {} {}", v0_idx, v1_idx, v2_idx)?;
                }
                Ok(())
            }

            /// Import a mesh from an .obj file Only work with triangles.
            pub fn import_from_obj_tri_only(
                file_path: &str,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                let (models, _materials) =
                    tobj::load_obj(file_path, &tobj::LoadOptions::default())?;
                let mut vertices = Vec::new();
                let mut triangles = Vec::new();

                for model in models {
                    let mesh = model.mesh;

                    // Add vertices
                    for i in 0..mesh.positions.len() / 3 {
                        vertices.push(Vertex {
                            x: mesh.positions[3 * i] as f64,
                            y: mesh.positions[3 * i + 1] as f64,
                            z: mesh.positions[3 * i + 2] as f64,
                        });
                    }

                    // Add triangles
                    for i in 0..mesh.indices.len() / 3 {
                        let v0 = vertices[mesh.indices[3 * i] as usize];
                        let v1 = vertices[mesh.indices[3 * i + 1] as usize];
                        let v2 = vertices[mesh.indices[3 * i + 2] as usize];

                        triangles.push(Triangle::new(v0, v1, v2));
                    }
                }
                Ok(Self {
                    vertices,
                    triangles,
                })
            }

            pub fn import_from_obj(file_path: &str) -> io::Result<Self> {
                let file = File::open(file_path)?;
                let reader = io::BufReader::new(file);

                let mut vertices: Vec<Vertex> = Vec::new();
                let mut triangles: Vec<Triangle> = Vec::new();

                for line in reader.lines() {
                    let line = line?;
                    let parts: Vec<&str> = line.split_whitespace().collect();

                    if parts.is_empty() {
                        continue; // Skip empty lines
                    }

                    match parts[0] {
                        "v" => {
                            // Parse a vertex
                            let x: f64 = parts[1].parse().expect("Invalid vertex format");
                            let y: f64 = parts[2].parse().expect("Invalid vertex format");
                            let z: f64 = parts[3].parse().expect("Invalid vertex format");
                            vertices.push(Vertex::new(x, y, z));
                        }
                        "f" => {
                            // Parse a face (convert quads to triangles)
                            let indices: Vec<usize> = parts[1..]
                                .iter()
                                .map(|s| s.split('/').next().unwrap().parse::<usize>().unwrap() - 1)
                                .collect();

                            if indices.len() == 3 {
                                // Triangle face
                                triangles.push(Triangle::new(
                                    vertices[indices[0]],
                                    vertices[indices[1]],
                                    vertices[indices[2]],
                                ));
                            } else if indices.len() == 4 {
                                // Quad face, split into two triangles
                                triangles.push(Triangle::new(
                                    vertices[indices[0]],
                                    vertices[indices[1]],
                                    vertices[indices[2]],
                                ));
                                triangles.push(Triangle::new(
                                    vertices[indices[0]],
                                    vertices[indices[2]],
                                    vertices[indices[3]],
                                ));
                            } else {
                                panic!("Unsupported face format: more than 4 vertices per face.");
                            }
                        }
                        _ => {
                            // Ignore other lines
                        }
                    }
                }
                Ok(Mesh::new(vertices, triangles))
            }

            /// an efficient way to import obj files.
            pub fn import_obj_with_normals(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
                use std::fs::File;
                use std::io::{BufRead, BufReader};

                let file = File::open(path)?;
                let reader = BufReader::new(file);

                let mut vertices = Vec::new();
                let mut normals = Vec::new();
                let mut triangles = Vec::new();

                for line in reader.lines() {
                    let line = line?;
                    let parts: Vec<&str> = line.split_whitespace().collect();

                    if parts.is_empty() {
                        continue;
                    }

                    match parts[0] {
                        "v" => {
                            // Vertex position
                            let x: f64 = parts[1].parse()?;
                            let y: f64 = parts[2].parse()?;
                            let z: f64 = parts[3].parse()?;
                            vertices.push(Vertex::new(x, y, z));
                        }
                        "vn" => {
                            // Vertex normal
                            let x: f64 = parts[1].parse()?;
                            let y: f64 = parts[2].parse()?;
                            let z: f64 = parts[3].parse()?;
                            normals.push(Vertex::new(x, y, z).unitize());
                        }
                        "f" => {
                            // Face
                            let mut face_vertices = Vec::new();
                            let mut face_normals = Vec::new();

                            for part in &parts[1..] {
                                let indices: Vec<&str> = part.split('/').collect();
                                let vertex_idx: usize = indices[0].parse::<usize>()? - 1; // .obj is 1-indexed
                                face_vertices.push(vertices[vertex_idx]);

                                // If normals are available
                                if indices.len() > 2 && !indices[2].is_empty() {
                                    let normal_idx: usize = indices[2].parse::<usize>()? - 1;
                                    face_normals.push(normals[normal_idx]);
                                }
                            }

                            if face_vertices.len() == 3 {
                                if face_normals.len() == 3 {
                                    // Use the first normal for the entire triangle
                                    triangles.push(Triangle::with_normal(
                                        face_vertices[0],
                                        face_vertices[1],
                                        face_vertices[2],
                                        face_normals[0],
                                    ));
                                } else {
                                    // Compute normal if not provided
                                    triangles.push(Triangle::new(
                                        face_vertices[0],
                                        face_vertices[1],
                                        face_vertices[2],
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Mesh {
                    vertices,
                    triangles,
                })
            }

            pub fn export_to_obj_with_normals(
                &self,
                path: &str,
            ) -> Result<(), Box<dyn std::error::Error>> {
                use std::fs::File;
                use std::io::{BufWriter, Write};

                let file = File::create(path)?;
                let mut writer = BufWriter::new(file);
                println!("\x1b[2J");
                println!("\x1b[3;0HExporting (speed not optimized) Path:({0})", path);
                let vertex_count = self.vertices.len();
                let mut ct = 0;
                for vertex in &self.vertices {
                    writeln!(writer, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
                    ct += 1;
                    println!("\x1b[4;0HVertex step Progress:{0}/{1}", ct, vertex_count);
                }

                let triangle_count = self.triangles.len();
                ct = 0;
                for triangle in &self.triangles {
                    writeln!(
                        writer,
                        "vn {} {} {}",
                        triangle.normal.x, triangle.normal.y, triangle.normal.z
                    )?;
                    ct += 1;
                    println!(
                        "\x1b[5;0HVertex Normals step Progress:{0}/{1}",
                        ct, triangle_count
                    );
                }
                ct = 0;
                for triangle in &self.triangles {
                    let v0_idx = self
                        .vertices
                        .iter()
                        .position(|v| *v == triangle.v0)
                        .unwrap()
                        + 1;
                    let v1_idx = self
                        .vertices
                        .iter()
                        .position(|v| *v == triangle.v1)
                        .unwrap()
                        + 1;
                    let v2_idx = self
                        .vertices
                        .iter()
                        .position(|v| *v == triangle.v2)
                        .unwrap()
                        + 1;
                    writeln!(
                        writer,
                        "f {}/{} {}/{} {}/{}",
                        v0_idx, v0_idx, v1_idx, v1_idx, v2_idx, v2_idx
                    )?;
                    ct += 1;
                    println!("\x1b[6;0HFace(s) step Progress:{0}/{1}", ct, triangle_count);
                }
                Ok(())
            }

            pub fn export_to_obj_with_normals_fast(
                &self,
                path: &str,
            ) -> Result<(), Box<dyn std::error::Error>> {
                let file = File::create(path)?;
                let mut writer = BufWriter::new(file);

                println!("\x1b[2J");
                println!("\x1b[3;0HExporting (optimized) Path:({})", path);

                // Step 1: Precompute vertex-to-index mapping
                let vertex_index_map: HashMap<&Vertex, usize> = self
                    .vertices
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v, i + 1)) // OBJ indices are 1-based
                    .collect();

                let vertex_count = self.vertices.len();
                let triangle_count = self.triangles.len();

                // Step 2: Generate vertex data in parallel
                println!("\x1b[4;0HGenerating vertex data...");
                let vertex_data: Vec<String> = self
                    .vertices
                    .par_iter()
                    .map(|vertex| format!("v {} {} {}\n", vertex.x, vertex.y, vertex.z))
                    .collect();

                // Step 3: Generate normal data in parallel
                println!("\x1b[5;0HGenerating normal data...");
                let normal_data: Vec<String> = self
                    .triangles
                    .par_iter()
                    .map(|triangle| {
                        format!(
                            "vn {} {} {}\n",
                            triangle.normal.x, triangle.normal.y, triangle.normal.z
                        )
                    })
                    .collect();

                // Step 4: Generate face data in parallel
                println!("\x1b[6;0HGenerating face data...");
                let face_data: Vec<String> = self
                    .triangles
                    .par_iter()
                    .map(|triangle| {
                        let v0_idx = *vertex_index_map.get(&triangle.v0).unwrap();
                        let v1_idx = *vertex_index_map.get(&triangle.v1).unwrap();
                        let v2_idx = *vertex_index_map.get(&triangle.v2).unwrap();
                        format!(
                            "f {}/{} {}/{} {}/{}\n",
                            v0_idx, v0_idx, v1_idx, v1_idx, v2_idx, v2_idx
                        )
                    })
                    .collect();

                // Step 5: Write data to file in batches
                println!("\x1b[7;0HWriting data to file...");
                for (i, line) in vertex_data.iter().enumerate() {
                    writer.write_all(line.as_bytes())?;
                    if i % 1000 == 0 || i == vertex_count - 1 {
                        println!("\x1b[4;0HVertex step Progress: {}/{}", i + 1, vertex_count);
                    }
                }

                for (i, line) in normal_data.iter().enumerate() {
                    writer.write_all(line.as_bytes())?;
                    if i % 1000 == 0 || i == triangle_count - 1 {
                        println!(
                            "\x1b[5;0HVertex Normals step Progress: {}/{}",
                            i + 1,
                            triangle_count
                        );
                    }
                }

                for (i, line) in face_data.iter().enumerate() {
                    writer.write_all(line.as_bytes())?;
                    if i % 1000 == 0 || i == triangle_count - 1 {
                        println!(
                            "\x1b[6;0HFace(s) step Progress: {}/{}",
                            i + 1,
                            triangle_count
                        );
                    }
                }

                println!("\x1b[8;0HExport completed successfully!");
                Ok(())
            }
        }

        impl Triangle {
            // Möller–Trumbore algorithm.
            pub fn intersect(&self, ray: &Ray) -> Option<f64> {
                let edge1 = Vertex {
                    x: self.v1.x - self.v0.x,
                    y: self.v1.y - self.v0.y,
                    z: self.v1.z - self.v0.z,
                };
                let edge2 = Vertex {
                    x: self.v2.x - self.v0.x,
                    y: self.v2.y - self.v0.y,
                    z: self.v2.z - self.v0.z,
                };

                let h = Vertex {
                    x: ray.direction.y * edge2.z - ray.direction.z * edge2.y,
                    y: ray.direction.z * edge2.x - ray.direction.x * edge2.z,
                    z: ray.direction.x * edge2.y - ray.direction.y * edge2.x,
                };
                let a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

                if a > -1e-8 && a < 1e-8 {
                    return None; // Ray is parallel to the triangle.
                }

                let f = 1.0 / a;
                let s = Vertex {
                    x: ray.origin.x - self.v0.x,
                    y: ray.origin.y - self.v0.y,
                    z: ray.origin.z - self.v0.z,
                };
                let u = f * (s.x * h.x + s.y * h.y + s.z * h.z);

                if u < 0.0 || u > 1.0 {
                    return None;
                }

                let q = Vertex {
                    x: s.y * edge1.z - s.z * edge1.y,
                    y: s.z * edge1.x - s.x * edge1.z,
                    z: s.x * edge1.y - s.y * edge1.x,
                };
                let v = f * (ray.direction.x * q.x + ray.direction.y * q.y + ray.direction.z * q.z);

                if v < 0.0 || u + v > 1.0 {
                    return None;
                }

                let t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

                if t > 1e-8 {
                    Some(t) // Intersection distance
                } else {
                    None
                }
            }

            // Compute the centroid of the triangle
            pub fn center(&self) -> [f64; 3] {
                let centroid = self.v0.add(&self.v1).add(&self.v2).div(3.0);
                [centroid.x, centroid.y, centroid.z]
            }

            // Compute the bounding box of the triangle
            pub fn bounding_box(&self) -> AABB {
                AABB {
                    min: Vertex {
                        x: self.v0.x.min(self.v1.x).min(self.v2.x),
                        y: self.v0.y.min(self.v1.y).min(self.v2.y),
                        z: self.v0.z.min(self.v1.z).min(self.v2.z),
                    },
                    max: Vertex {
                        x: self.v0.x.max(self.v1.x).max(self.v2.x),
                        y: self.v0.y.max(self.v1.y).max(self.v2.y),
                        z: self.v0.z.max(self.v1.z).max(self.v2.z),
                    },
                }
            }
        }

        #[derive(Debug, Clone)]
        pub struct Ray {
            pub origin: Vertex,
            pub direction: Vertex,
        }

        impl Ray {
            pub fn new(pt1: Vertex, pt2: Vertex) -> Self {
                Self {
                    origin: pt1,
                    direction: pt2,
                }
            }
        }

        // A Bounding Volume Hierarchy (BVH) organizes objects (e.g., triangles)
        // into a tree structure to accelerate ray tracing
        // by reducing the number of intersection tests.

        // AABB stand for Axis aligned Bounding Box.
        #[derive(Debug, Clone)]
        pub struct AABB {
            min: Vertex, // Minimum corner of the bounding box
            max: Vertex, // Maximum corner of the bounding box
        }

        impl AABB {
            // Combine two AABBs into one that encompasses both
            pub fn surrounding_box(box1: &AABB, box2: &AABB) -> AABB {
                AABB {
                    min: Vertex {
                        x: box1.min.x.min(box2.min.x),
                        y: box1.min.y.min(box2.min.y),
                        z: box1.min.z.min(box2.min.z),
                    },
                    max: Vertex {
                        x: box1.max.x.max(box2.max.x),
                        y: box1.max.y.max(box2.max.y),
                        z: box1.max.z.max(box2.max.z),
                    },
                }
            }

            // Ray-AABB intersection test (needed for BVH traversal)
            pub fn intersects(&self, ray: &Ray) -> bool {
                let inv_dir = Vertex {
                    x: 1.0 / ray.direction.x,
                    y: 1.0 / ray.direction.y,
                    z: 1.0 / ray.direction.z,
                };

                let t_min = (
                    (self.min.x - ray.origin.x) * inv_dir.x,
                    (self.min.y - ray.origin.y) * inv_dir.y,
                    (self.min.z - ray.origin.z) * inv_dir.z,
                );
                let t_max = (
                    (self.max.x - ray.origin.x) * inv_dir.x,
                    (self.max.y - ray.origin.y) * inv_dir.y,
                    (self.max.z - ray.origin.z) * inv_dir.z,
                );

                let t_enter = t_min.0.max(t_min.1).max(t_min.2);
                let t_exit = t_max.0.min(t_max.1).min(t_max.2);

                t_enter <= t_exit
            }
        }

        #[derive(Debug)]
        pub enum BVHNode {
            Leaf {
                bounding_box: AABB,
                triangles: Vec<Triangle>, // Triangles in the leaf
            },
            Internal {
                bounding_box: AABB,
                left: Arc<BVHNode>,  // Left child
                right: Arc<BVHNode>, // Right child
            },
        }

        impl BVHNode {
            // A Bounding Volume Hierarchy (BVH) organizes objects (e.g., triangles)
            // into a tree structure to accelerate ray tracing by
            // reducing the number of intersection tests.
            pub fn build(triangles: Vec<Triangle>, depth: usize) -> BVHNode {
                // Base case: Create a leaf node if triangle count is small
                if triangles.len() <= 2 {
                    let bounding_box = triangles
                        .iter()
                        .map(|tri| tri.bounding_box())
                        .reduce(|a, b| AABB::surrounding_box(&a, &b))
                        .unwrap();
                    return BVHNode::Leaf {
                        bounding_box,
                        triangles,
                    };
                }

                // Find the axis to split (X, Y, or Z)
                let axis = depth % 3;
                let mut sorted_triangles = triangles;
                sorted_triangles.sort_by(|a, b| {
                    let center_a = a.center()[axis];
                    let center_b = b.center()[axis];
                    center_a.partial_cmp(&center_b).unwrap()
                });

                // Partition the triangles into two groups
                let mid = sorted_triangles.len() / 2;
                let (left_triangles, right_triangles) = sorted_triangles.split_at(mid);

                // Recursively build the left and right subtrees
                let left = Arc::new(BVHNode::build(left_triangles.to_vec(), depth + 1));
                let right = Arc::new(BVHNode::build(right_triangles.to_vec(), depth + 1));

                // Create the bounding box for this node
                let bounding_box =
                    AABB::surrounding_box(&left.bounding_box(), &right.bounding_box());

                BVHNode::Internal {
                    bounding_box,
                    left,
                    right,
                }
            }

            pub fn bounding_box(&self) -> &AABB {
                match self {
                    BVHNode::Leaf { bounding_box, .. } => bounding_box,
                    BVHNode::Internal { bounding_box, .. } => bounding_box,
                }
            }
            pub fn intersect(&self, ray: &Ray) -> Option<(f64, &Triangle)> {
                if !self.bounding_box().intersects(ray) {
                    return None; // Ray doesn't hit this node
                }

                match self {
                    BVHNode::Leaf { triangles, .. } => {
                        // Test against all triangles in the leaf
                        let mut closest_hit: Option<(f64, &Triangle)> = None;
                        for triangle in triangles {
                            if let Some(t) = triangle.intersect(ray) {
                                if closest_hit.is_none() || t < closest_hit.unwrap().0 {
                                    closest_hit = Some((t, triangle));
                                }
                            }
                        }
                        closest_hit
                    }
                    BVHNode::Internal { left, right, .. } => {
                        // Recursively test left and right children
                        let left_hit = left.intersect(ray);
                        let right_hit = right.intersect(ray);

                        match (left_hit, right_hit) {
                            (Some(l), Some(r)) => {
                                if l.0 < r.0 {
                                    Some(l)
                                } else {
                                    Some(r)
                                }
                            }
                            (Some(l), None) => Some(l),
                            (None, Some(r)) => Some(r),
                            (None, None) => None,
                        }
                    }
                }
            }
        }

        pub struct Camera {
            position: Vertex,  // Camera position
            forward: Vertex,   // Forward direction
            right: Vertex,     // Right direction
            up: Vertex,        // Up direction
            fov: f64,          // Field of view in degrees
            aspect_ratio: f64, // Aspect ratio (width / height)
            width: usize,      // Image width
            height: usize,     // Image height
        }

        impl Camera {
            // Generate rays for a given pixel
            pub fn generate_ray(&self, pixel_x: usize, pixel_y: usize) -> Ray {
                // Convert FOV to radians and compute scaling factor
                let fov_scale = (self.fov.to_radians() / 2.0).tan();

                // Map pixel to normalized device coordinates (NDC)
                let ndc_x = (pixel_x as f64 + 0.5) / self.width as f64; // Center pixel
                let ndc_y = (pixel_y as f64 + 0.5) / self.height as f64;

                // Map NDC to screen space [-1, 1]
                let screen_x = 2.0 * ndc_x - 1.0;
                let screen_y = 1.0 - 2.0 * ndc_y;

                // Adjust for aspect ratio and FOV
                let pixel_camera_x = screen_x * self.aspect_ratio * fov_scale;
                let pixel_camera_y = screen_y * fov_scale;

                // Compute ray direction in camera space
                let ray_direction = self
                    .forward
                    .add(&self.right.mul(pixel_camera_x))
                    .add(&self.up.mul(pixel_camera_y))
                    .unitize(); // Normalize to get unit vector

                // Create the ray
                Ray {
                    origin: self.position,
                    direction: ray_direction,
                }
            }

            // Generate a memory allocated array: Vec<Ray> of Ray object for further
            // Ray tracing from Camera fov generated rays.
            pub fn render(camera: &Camera, ray_buffer: &mut Vec<Ray>) {
                for y in 0..camera.height {
                    for x in 0..camera.width {
                        let ray = camera.generate_ray(x, y);
                        println!(
                            "Pixel ({}, {}) -> Ray Origin: {:?}, Direction: {:?}",
                            x, y, ray.origin, ray.direction
                        );
                        ray_buffer.push(ray);
                    }
                }
            }
        }
        mod ray_operations {
            fn shade_with_distance(
                base_color: (u8, u8, u8),
                distance: f64,
                attenuation: f64,
            ) -> (u8, u8, u8) {
                let intensity = 1.0 / (1.0 + attenuation * distance);
                let (r, g, b) = base_color;

                (
                    (r as f64 * intensity) as u8,
                    (g as f64 * intensity) as u8,
                    (b as f64 * intensity) as u8,
                )
            }
            fn fog_with_distance(
                base_color: (u8, u8, u8),
                fog_color: (u8, u8, u8),
                distance: f64,
                max_distance: f64,
            ) -> (u8, u8, u8) {
                let fog_factor = (distance / max_distance).min(1.0);
                let (r1, g1, b1) = base_color;
                let (r2, g2, b2) = fog_color;

                (
                    ((r1 as f64 * (1.0 - fog_factor)) + (r2 as f64 * fog_factor)) as u8,
                    ((g1 as f64 * (1.0 - fog_factor)) + (g2 as f64 * fog_factor)) as u8,
                    ((b1 as f64 * (1.0 - fog_factor)) + (b2 as f64 * fog_factor)) as u8,
                )
            }
            fn light_falloff(base_color: (u8, u8, u8), distance: f64) -> (u8, u8, u8) {
                let intensity = 1.0 / (distance + 1.0); // Soft falloff
                let (r, g, b) = base_color;

                (
                    (r as f64 * intensity) as u8,
                    (g as f64 * intensity) as u8,
                    (b as f64 * intensity) as u8,
                )
            }
        }
    }
    pub mod coloring {
        /*
         * Color struct hold the RGB values in 3, 8 bit values and the total
         * absolute color value in a single u32 bit value
         * Most of the time expressed in Hexadecimal OxFFFFFFFF (4x8bit)
         * by other API.
         * - the absolute value and back ground value are optionals
         *   and can be cached for accelerated process.
         * - alpha is set to 1.0 (opaque) if not needed.
         * - every components are automatically cached and updated if
         *   they are claimed by runtime.
         * - everything is always kept Updated since structure is private
         *   and use only getter and setter as unique way to update
         *   the color description.
         * - alpha can be removed and RGB values are restored to the original
         *   opaque colors.
         * - draw back: it's a low level implementation and so alpha is always
         *   relative to a defined back ground color on function call.
         * */
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
                            (self.red as u32) << 16
                                | ((self.green as u32) << 8)
                                | (self.blue as u32),
                        );
                        // Update RGB description components from updated absolute value.
                        self.red = ((self.value.unwrap() >> 16) & 0xFF) as u8;
                        self.green = ((self.value.unwrap() >> 8) & 0xFF) as u8;
                        self.blue = ((self.value.unwrap()) & 0xFF) as u8;
                        self.value.unwrap() // return the computed absolute value.
                    } else {
                        // Update absolute value from RGB.
                        self.value = Some(
                            (self.red as u32) << 16
                                | ((self.green as u32) << 8)
                                | (self.blue as u32),
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
                    self.value = Some(
                        (self.red as u32) >> 16 | (self.green as u32) >> 8 | (self.green as u32),
                    );
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

                //TODO: a struct for Background color may be usefull
                // for caching in u8 form ....
                // Extract background RGB components as f32.
                let bg_r = ((bg_color >> 16) & 0xFF) as f32;
                let bg_g = ((bg_color >> 8) & 0xFF) as f32;
                let bg_b = (bg_color & 0xFF) as f32;

                // Blend each channel
                let blended_r = ((*alpha) * (*red) as f32 + (1.0 - (*alpha)) * bg_r).round() as u32;
                let blended_g =
                    ((*alpha) * (*green) as f32 + (1.0 - (*alpha)) * bg_g).round() as u32;
                let blended_b =
                    ((*alpha) * (*blue) as f32 + (1.0 - (*alpha)) * bg_b).round() as u32;
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
        }
    }
}

/*
 *  A set of early very basic transformations method
 *  of Point3d from world axis and Angles.
 */
pub mod transformation {

    use super::geometry::Point3d;
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

    pub fn rotate_x(point: Point3d, angle: f64) -> Point3d {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Point3d {
            X: point.X,
            Y: point.Y * cos_theta - point.Z * sin_theta,
            Z: point.Y * sin_theta + point.Z * cos_theta,
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

    use crate::models_3d::FONT_5X7;
    use super::geometry::Point3d;

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
        let grid_unit =  grid_spacing_unit / x_max;
        let mut x = 0.0;
        let mut y = 0.0;
        while x <= *x_max {
            while y <= *y_max {
                grid_points.push((*plane).point_on_plane_uv(&x, &y));
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
    pub fn draw_3d_circle(origin:Point3d,radius:f64,step:f64)->Vec<Point3d>{
        let mut increm = 0.0f64;
        let mut circle_pts = Vec::new();
        while  increm <= (f64::consts::PI*2.0){
            circle_pts.push(Point3d::new((f64::sin(increm)*radius)+origin.X,(f64::cos(increm)*radius)+origin.Y,0.0+origin.Z));
            increm += (f64::consts::PI*2.0)/step;
        }
        circle_pts
    } 
}

pub mod utillity {
    use core::f64;
    pub fn degree_to_radians(inputangle_in_degre: &f64) -> f64 {
        (*inputangle_in_degre) * (f64::consts::PI * 2.0) / 360.0
    }
}

#[cfg(test)]
mod test {
    use crate::{project_3d_point_on_plane, redering_object};

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
        let result = cp.point_on_plane(&(point.X), &(point.Y), &(point.Z));
        // assert_eq!(expected_result,result);
        if (expected_result - result).Length().abs() <= 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_degrees_to_radian() {
        use super::utillity::*;
        let angle_to_test = 90.0;
        assert_eq!(f64::consts::PI / 2.0, degree_to_radians(&angle_to_test));
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
        use super::visualization::redering_object::*;

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

    use super::visualization::coloring::Color;
    #[test]
    fn test_color() {
        let red: u8 = 20;
        let green: u8 = 19;
        let blue: u8 = 20;
        assert_eq!(0x141314, Color::convert_rgb_color(red, green, blue));
    }
    use super::visualization::redering_object::*;
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
}
