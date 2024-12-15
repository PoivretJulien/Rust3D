// ************************************************************************
// ******* First scratch of a basic computational component class *********
// ************************************************************************
mod rust_3d {
    pub mod Geometry {
        // Implementation of a Point3d structure
        // bound to Vector3d structure
        // for standard operator processing.
        use std::ops::{Add, Div, Mul, MulAssign, Sub};
        #[allow(non_snake_case)]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct Point3d {
            pub X: f64,
            pub Y: f64,
            pub Z: f64,
        }

        #[allow(non_snake_case)]
        impl Point3d {
            /// Create a 3d point.
            ///  # Arguments
            ///  (x:f64, y:f64, z:f64)
            ///  # Returns
            ///  a new Point3d from x,y,z values.
            pub fn new(x: f64, y: f64, z: f64) -> Self {
                Self { X: x, Y: y, Z: z }
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
            ///  ( the vector Length is automatically
            ///  computed at vector creation ).
            pub fn new(x: f64, y: f64, z: f64) -> Self {
                Self {
                    X: x,
                    Y: y,
                    Z: z,
                    Length: Vector3d::compute_vector_length_b(x, y, z),
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
            pub fn compute_vector_length_b(x: f64, y: f64, z: f64) -> f64 {
                (x.powi(2) + y.powi(2) + z.powi(2)).sqrt()
            }

            // Vector3d Cross Product.
            pub fn cross_product(vector_a: Vector3d, vector_b: Vector3d) -> Self {
                Vector3d::new(
                    vector_a.Y * vector_b.Z - vector_a.Z * vector_b.Y,
                    vector_a.Z * vector_b.X - vector_a.X * vector_b.Z,
                    vector_a.X * vector_b.Y - vector_a.Y * vector_b.X,
                )
            }

            // Unitize the Vector3d.
            pub fn unitize(&mut self) {
                self.X /= self.Length;
                self.Y /= self.Length;
                self.Z /= self.Length;
                self.update_length();
            }

            pub fn unitize_b(&mut self) -> Vector3d {
                Vector3d::new(
                    self.X / self.Length,
                    self.Y / self.Length,
                    self.Z / self.Length,
                )
            }

            /// return the angle between two vectors
            pub fn compute_angle(vector_a: Vector3d, vector_b: Vector3d) -> f64 {
                f64::acos(
                    (vector_a * vector_b)
                        / (f64::sqrt(vector_a.X.powi(2) + vector_a.Y.powi(2) + vector_a.Z.powi(2))
                            * f64::sqrt(
                                vector_b.X.powi(2) + vector_b.Y.powi(2) + vector_b.Z.powi(2),
                            )),
                )
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

        impl Mul for Vector3d {
            type Output = f64;
            fn mul(self, vector: Vector3d) -> f64 {
                self.X * vector.X + self.Y * vector.Y + self.Z * vector.Z
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
    }
    
    /*
     * 'atomic' Case study for a simple representation of a 3d point 
     *  on simple 2D screen via projection matrix honestly for now it's kind of 
     *  a magic black box for me but let's refine the analysis.
    */
    pub mod visualization {

        use super::Geometry::{Point3d, Vector3d};
        
        struct Camera {
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
            fn new(
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

            // Compute the view matrix
            // (transforms world coordinates to camera coordinates)
            fn get_view_matrix(&self) -> [[f64; 4]; 4] {
                let forward = Vector3d::new(
                    self.target.get_X() - self.position.get_X(),
                    self.target.get_Y() - self.position.get_Y(),
                    self.target.get_Z() - self.position.get_Z(),
                )
                .unitize_b();
                let right = Vector3d::cross_product(forward, self.up).unitize_b();
                let up = Vector3d::cross_product(right, forward).unitize_b();
                let translation = Vector3d::new(
                    -self.position.get_X(),
                    -self.position.get_Y(),
                    -self.position.get_Z(),
                );
                [
                    [right.get_X(), up.get_X(), -forward.get_X(), 0.0],
                    [right.get_Y(), up.get_Y(), -forward.get_Y(), 0.0],
                    [right.get_Z(), up.get_Z(), -forward.get_Z(), 0.0],
                    [
                        right * translation,
                        up * translation,
                        forward * translation,
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
            fn project(&self, point: Point3d) -> Option<(usize, usize)> {
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
}

#[cfg(test)]
mod test {
    use super::rust_3d::Geometry::*;
    #[test]
    fn test_cross_product() {
        let vec_a = Vector3d::new(1.0, 0.0, 0.0);
        let vec_b = Vector3d::new(0.0, 1.0, 0.0);
        assert_eq!(
            Vector3d::new(0.0, 0.0, 1.0),
            Vector3d::cross_product(vec_a, vec_b)
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
    use std::f64::consts::PI;
    #[test]
    fn test_vector3d_angle() {
        assert_eq!(
            PI / 2.0,
            Vector3d::compute_angle(Vector3d::new(0.0, 1.0, 0.0), Vector3d::new(1.0, 0.0, 0.0))
        );
    }
}
