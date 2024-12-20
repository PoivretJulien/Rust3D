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
            self.Length = ((self.X.powi(2)) + (self.Y.powi(2)) + (self.Z.powi(2))).sqrt();
        }

        /// static way to compute vector length.
        /// # Arguments
        /// x:f 64, y:f 64, z:f 64
        /// # Returns
        /// return a f 64 length distance.
        pub fn compute_length(x: f64, y: f64, z: f64) -> f64 {
            (x.powi(2) + y.powi(2) + z.powi(2)).sqrt()
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

        /// return the angle between two vectors
        pub fn compute_angle(vector_a: &Vector3d, vector_b: &Vector3d) -> f64 {
            f64::acos(
                ((*vector_a) * (*vector_b))
                    / (f64::sqrt(
                        (*vector_a).X.powi(2) + (*vector_a).Y.powi(2) + (*vector_a).Z.powi(2),
                    ) * f64::sqrt(
                        (*vector_b).X.powi(2) + (*vector_b).Y.powi(2) + (*vector_b).Z.powi(2),
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
    pub fn intersect_line_with_plane(
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
        position: Vector3d, // Camera position in world space
        target: Vector3d,   // The point the camera is looking at
        up: Vector3d,       // The "up" direction (usually the Y-axis)
        fov: f64,           // Field of view (in degrees)
        width: f64,         // Screen width
        height: f64,        // Screen height
        near: f64,          // Near clipping plane
        far: f64,           // Far clipping plane
    }

    impl Camera {
        pub fn new(
            position: Vector3d,
            target: Vector3d,
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
                self.position.get_X() - self.target.get_X(),
                self.position.get_Y() - self.target.get_Y(),
                self.position.get_Z() - self.target.get_Z(),
            )
            .unitize_b();
            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();
            let up = Vector3d::cross_product(&right, &forward).unitize_b();
            // a Point3d is used there instead of Vector3d to avoid
            // computing unused vector length automatically.
            let translation = Point3d::new(
                -self.position.get_X(),
                -self.position.get_Y(),
                -self.position.get_Z(),
            );
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
    /// Project a 3d point plane.
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
}

pub mod utillity {
    use core::f64;
    pub fn Degree_to_Radians(inputangle_in_degre: &f64) -> f64 {
        (*inputangle_in_degre) * (f64::consts::PI * 2.0) / 360.0
    }
}

#[cfg(test)]
mod test {
    use crate::project_3d_point_on_plane;

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
    fn test_degrees_to_radiansi() {
        use super::utillity::*;
        let angle_to_test = 90.0;
        assert_eq!(f64::consts::PI / 2.0, Degree_to_Radians(&angle_to_test));
    }

    #[test]
    fn test_ray_trace() {
        use super::intersection::*;
        let point = Point3d::new(7.812578, 4.543698, 23.058283);
        let direction = Vector3d::new(-1.398849, 0.106953, -0.982613);
        let plane_origin = Point3d::new(-8.015905, -1.866453, 5.80651);
        let plane_normal = Vector3d::new(0.65694, -0.31293, 0.685934);
        let plane = CPlane::new(&plane_origin, &plane_normal);
        let expected_result = Point3d::new(-9.583205, 5.873738, 10.838721);
        // assert_eq!(expected_result,intersect_line_with_plane(&point, &direction, &plane).unwrap());
        if let Some(result_point) = intersect_line_with_plane(&point, &direction, &plane){
           if (result_point - expected_result).Length().abs() < 1e-5{
               assert!(true);
           }else{
               assert!(false);
           }
        }
    }
}
