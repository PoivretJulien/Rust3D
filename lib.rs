// ************************************************************************
// ******* First scratch of a basic computational component class *********
// ************************************************************************
#[allow(dead_code)]
mod rust_3d {
    // Implementation of Point3d structure
    // bound to vector 3d for standard operator processing.
    use std::ops::{Add, Sub};
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
        pub X: f64,
        pub Y: f64,
        pub Z: f64,
        pub Length: f64,
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
        // Some methods overload a&b.
        /// Compute the vector length.
        pub fn compute_vector_length_a(&mut self) {
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
}

#[cfg(test)]
mod test {
    use super::rust_3d::*;
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
    fn test_vector3d_length(){
        assert_eq!(f64::sqrt(2.0),Vector3d::new(1.0,1.0,0.0).Length);
    }
}
