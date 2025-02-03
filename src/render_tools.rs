/*
 *     Visualization V3 bring Vertex standardization
 *     and some code optimization providing matrix 4x4
 *     4x3 operations for Vertex translation and rotation.
 *     - perspective projection are always in 4x4
 *       (with Homogeneous transformation)
 *     - apply transformation matrix in parallel,
 *       on method with suffix _par
 */
pub mod visualization_v3 {
    use super::rendering_object::Vertex;
    use crate::rust3d::geometry::{Point3d, Vector3d};
    use rayon::prelude::*;

    pub struct Camera {
        pub position: Point3d,
        pub target: Point3d,
        pub up: Vector3d,
        pub fov: f64,
        pub width: f64,
        pub height: f64,
        pub near: f64,
        pub far: f64,
        pub view_matrix: [[f64; 4]; 4], // Precomputed view matrix
        pub projection_matrix: [[f64; 4]; 4], // Precomputed projection matrix
    }

    impl Camera {
        /// Construct a camera with cached matrix conversion
        /// which involve to update the matrix if one of the camera component
        /// has changed. use .update_matrices() on the camera object for that
        /// purpose.
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
            let mut camera = Self {
                position,
                target,
                up,
                fov,
                width,
                height,
                near,
                far,
                view_matrix: [[0.0; 4]; 4], // Temporary initialization
                projection_matrix: [[0.0; 4]; 4], // Temporary initialization
            };

            // Precompute the matrices
            camera.update_matrices();
            camera
        }

        /// Project a set of Vertex on 2d space.
        /// this is multitreaded.
        /// # Returns
        /// a vector of tuple (x,y,depth) (usize,usize,f64)
        pub fn project_points(&self, points: &Vec<Vertex>) -> Vec<(usize, usize, f64)> {
            points
                .par_iter() // Use parallel iterator
                .filter_map(|point| self.project(point)) // Apply the project method in parallel
                .collect() // Collect results into a Vec
        }

        /// Update the view and projection matrices (call when camera parameters change)
        pub fn update_matrices(&mut self) {
            self.view_matrix = self.compute_view_matrix();
            self.projection_matrix = self.compute_projection_matrix();
        }

        /// Compute the view matrix
        pub fn compute_view_matrix(&self) -> [[f64; 4]; 4] {
            let forward = Vector3d::new(
                self.position.X - self.target.X,
                self.position.Y - self.target.Y,
                self.position.Z - self.target.Z,
            )
            .unitize_b();
            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Vertex as Vector3d to avoid run time penalty.
            let translation = Vertex::new(-self.position.X, -self.position.Y, -self.position.Z);
            [
                [right.get_X(), up.get_X(), forward.get_X(), 0.0],
                [right.get_Y(), up.get_Y(), forward.get_Y(), 0.0],
                [right.get_Z(), up.get_Z(), forward.get_Z(), 0.0],
                [
                    (right.get_X() * translation.x)
                        + (right.get_Y() * translation.y)
                        + (right.get_Z() * translation.z),
                    (up.get_X() * translation.x)
                        + (up.get_Y() * translation.y)
                        + (right.get_Z() * translation.z),
                    (forward.get_X() * translation.x)
                        + (forward.get_Y() * translation.y)
                        + (forward.get_Z() * translation.z),
                    1.0,
                ],
            ]
        }

        /// Compute the projection matrix
        pub fn compute_projection_matrix(&self) -> [[f64; 4]; 4] {
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

        /// Project a vertex in a 2d camera space projection.
        pub fn project(&self, point: &Vertex) -> Option<(usize, usize, f64)> {
            // Use precomputed matrices
            let camera_space_point = self.multiply_matrix_vector(self.view_matrix, point);

            // Extract depth in camera space (before projection)
            let depth_in_camera_space = camera_space_point.z;

            let projected_point =
                self.multiply_matrix_vector(self.projection_matrix, &camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.x / projected_point.z;
            let y = projected_point.y / projected_point.z;

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

            Some((screen_x as usize, screen_y as usize, depth_in_camera_space))
        }

        #[inline(always)]
        /// Project a vertex in a 2d camera space projection.
        pub fn project_maybe_outside(&self, point: &Vertex) -> (f64, f64) {
            // Use precomputed matrices
            let camera_space_point = self.multiply_matrix_vector(self.view_matrix, point);

            let projected_point =
                self.multiply_matrix_vector(self.projection_matrix, &camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.x / projected_point.z;
            let y = projected_point.y / projected_point.z;

            // Map the coordinates from [-1, 1] to screen space
            let screen_x = ((x + 1.0) * 0.5 * self.width) as isize;
            let screen_y = ((1.0 - y) * 0.5 * self.height) as isize;

            (screen_x as f64, screen_y as f64)
        }

        pub fn multiply_matrix_vector(&self, matrix: [[f64; 4]; 4], v: &Vertex) -> Vertex {
            Vertex::new(
                matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z + matrix[0][3],
                matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z + matrix[1][3],
                matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z + matrix[2][3],
            )
        }

        /// Generate a transformation matrix for the camera's movement and rotation
        pub fn get_transformation_matrix(
            &self,
            forward_amount: f64,
            yaw_angle: f64,
            pitch_angle: f64,
        ) -> [[f64; 4]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vertex::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .normalize();

            // Step 2: Compute the translation vector for the camera movement
            let translation = forward * (-forward_amount);

            // Step 3: Compute rotation matrices
            // Yaw (rotation around the Y-axis)
            let yaw_cos = yaw_angle.to_radians().cos();
            let yaw_sin = yaw_angle.to_radians().sin();
            let yaw_matrix = [
                [yaw_cos, 0.0, yaw_sin, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-yaw_sin, 0.0, yaw_cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Pitch (rotation around the X-axis)
            let pitch_cos = pitch_angle.to_radians().cos();
            let pitch_sin = pitch_angle.to_radians().sin();
            let pitch_matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, pitch_cos, -pitch_sin, 0.0],
                [0.0, pitch_sin, pitch_cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Step 4: Combine yaw and pitch into a single rotation matrix
            let rotation_matrix = self.multiply_matrices(yaw_matrix, pitch_matrix);

            // Step 5: Create the translation matrix
            let translation_matrix = [
                [1.0, 0.0, 0.0, translation.x],
                [0.0, 1.0, 0.0, translation.y],
                [0.0, 0.0, 1.0, translation.z],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Step 6: Combine rotation and translation matrices into a single transformation matrix
            self.multiply_matrices(rotation_matrix, translation_matrix)
        }

        /// Apply a transformation matrix to a Vec<Vertex> and return the transformed points
        pub fn transform_points(
            &self,
            points: &Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) -> Vec<Vertex> {
            points
                .iter()
                .map(|point| self.multiply_matrix_vector(transformation_matrix, point))
                .collect()
        }
        /// Apply a transformation matrix to a Vec<Vertex> and return the transformed points
        /// parallelized version of the method.
        pub fn transform_points_par(
            &self,
            points: &Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) -> Vec<Vertex> {
            points
                .par_iter()
                .map(|point| self.multiply_matrix_vector(transformation_matrix, point))
                .collect()
        }

        /// Apply a transformation matrix to a mutable Vec<Vertex>
        pub fn transform_points_mut(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) {
            points.iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector(transformation_matrix, point);
            });
        }
        /// Apply a transformation matrix to a mutable Vec<Vertex>
        /// parallelized version of the method.
        pub fn transform_points_mut_par(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) {
            points.par_iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector(transformation_matrix, point);
            });
        }
        /// Generate a transformation matrix for panning the camera
        pub fn get_pan_matrix(&self, right_amount: f64, up_amount: f64) -> [[f64; 4]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vector3d::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .unitize_b();

            // Step 2: Compute the right direction vector (cross product of forward and up)
            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();

            // Step 3: Compute the true up direction (orthogonalized)
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Step 4: Scale the right and up vectors by the panning amounts
            let pan_translation =
                right.to_vertex() * (-right_amount) + (up.to_vertex() * up_amount);

            // Step 5: Create the translation matrix for panning
            [
                [1.0, 0.0, 0.0, pan_translation.x],
                [0.0, 1.0, 0.0, pan_translation.y],
                [0.0, 0.0, 1.0, pan_translation.z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }

        /// Apply panning to a set of Vertex.
        pub fn pan_points(
            &self,
            points: &Vec<Vertex>,
            right_amount: f64,
            up_amount: f64,
        ) -> Vec<Vertex> {
            // Step 1: Get the pan transformation matrix
            let pan_matrix = self.get_pan_matrix(right_amount, up_amount);

            // Step 2: Apply the pan transformation to all points
            points
                .iter()
                .map(|point| self.multiply_matrix_vector(pan_matrix, point))
                .collect()
        }

        /// Apply panning to a set of mutable Vertex points.
        pub fn pan_points_mut(&self, points: &mut Vec<Vertex>, right_amount: f64, up_amount: f64) {
            // Step 1: Get the pan transformation matrix
            let pan_matrix = self.get_pan_matrix(right_amount, up_amount);

            // Step 2: Apply the pan transformation to all points
            points.iter_mut().for_each(|point| {
                (*point) = self.multiply_matrix_vector(pan_matrix, point);
            });
        }

        /// Generate a panning transformation matrix
        /// `dx` and `dy` are the offsets in world space along the right and up directions.
        pub fn pan_point_matrix(&self, dx: f64, dy: f64) -> [[f64; 4]; 4] {
            // Calculate the right and up vectors based on the camera's orientation
            let forward = Vector3d::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .unitize_b();

            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Translation in the right and up directions
            let translation = Vertex::new(
                dx * right.get_X() + dy * up.get_X(),
                dx * right.get_Y() + dy * up.get_Y(),
                dx * right.get_Z() + dy * up.get_Z(),
            );

            // Construct the transformation matrix
            [
                [1.0, 0.0, 0.0, translation.x],
                [0.0, 1.0, 0.0, translation.y],
                [0.0, 0.0, 1.0, translation.z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }

        /// Create a rotation matrix from angles (in degrees) for X, Y, and Z axes
        pub fn rotation_matrix_from_angles(
            x_angle: f64,
            y_angle: f64,
            z_angle: f64,
        ) -> [[f64; 4]; 4] {
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
            let rotation_xy = Self::multiply_matrices_axb(rotation_y, rotation_x);
            let rotation_xyz = Self::multiply_matrices_axb(rotation_z, rotation_xy);

            rotation_xyz
        }

        /// Combine multiple transformation matrices into one
        /// they have to be call from a stack vector use macro vec! for that.
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
                result = Self::multiply_matrices_axb(result, matrix);
            }

            result
        }

        /// Helper function to multiply two 4x4 matrices
        pub fn multiply_matrices_axb(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
            let mut result = [[0.0; 4]; 4];

            for i in 0..4 {
                for j in 0..4 {
                    result[i][j] = a[i][0] * b[0][j]
                        + a[i][1] * b[1][j]
                        + a[i][2] * b[2][j]
                        + a[i][3] * b[3][j];
                }
            }
            result
        }
        /// Utility function: Multiply two 4x4 matrices (non static method version).
        fn multiply_matrices(&self, a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
            let mut result = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    result[i][j] = a[i][0] * b[0][j]
                        + a[i][1] * b[1][j]
                        + a[i][2] * b[2][j]
                        + a[i][3] * b[3][j];
                }
            }
            result
        }
        // 4x3 matrix version of the code, the absance of the Homogeneous Normalization
        // modify the sensitivity of the system.
        // but provide a boost in computation performance
        // (from the drop of the last row of the 4x4 matrix)
        // use regular 4x4 version if you preferes more precision in motion.
        // - all 4x3 operations are explicitly named with _4x3 sufix.
        /// Generate a transformation matrix for the camera's movement and rotation
        pub fn get_transformation_matrix_4x3(
            &self,
            forward_amount: f64,
            yaw_angle: f64,
            pitch_angle: f64,
        ) -> [[f64; 3]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vertex::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .normalize();

            // Step 2: Compute the translation vector for the camera movement
            let translation = forward * (-forward_amount);

            // Step 3: Compute rotation matrices
            // Yaw (rotation around the Y-axis)
            let yaw_cos = yaw_angle.to_radians().cos();
            let yaw_sin = yaw_angle.to_radians().sin();
            let yaw_matrix = [
                [yaw_cos, 0.0, yaw_sin],
                [0.0, 1.0, 0.0],
                [-yaw_sin, 0.0, yaw_cos],
            ];

            // Pitch (rotation around the X-axis)
            let pitch_cos = pitch_angle.to_radians().cos();
            let pitch_sin = pitch_angle.to_radians().sin();
            let pitch_matrix = [
                [1.0, 0.0, 0.0],
                [0.0, pitch_cos, -pitch_sin],
                [0.0, pitch_sin, pitch_cos],
            ];

            // Step 4: Combine yaw and pitch into a single rotation matrix
            let rotation_matrix = self.multiply_matrices_3x3(yaw_matrix, pitch_matrix);

            // Step 5: Combine the rotation matrix and translation into a single 4x3 transformation matrix
            [
                [
                    rotation_matrix[0][0],
                    rotation_matrix[0][1],
                    rotation_matrix[0][2],
                ],
                [
                    rotation_matrix[1][0],
                    rotation_matrix[1][1],
                    rotation_matrix[1][2],
                ],
                [
                    rotation_matrix[2][0],
                    rotation_matrix[2][1],
                    rotation_matrix[2][2],
                ],
                [translation.x, translation.y, translation.z],
            ]
        }

        /// Apply a transformation matrix to a Vec<Vertex> and return the transformed points
        pub fn transform_points_4x3(
            &self,
            points: &Vec<Vertex>,
            transformation_matrix: [[f64; 3]; 4],
        ) -> Vec<Vertex> {
            points
                .iter()
                .map(|point| self.multiply_matrix_vector_4x3(transformation_matrix, *point))
                .collect()
        }

        /// Apply a transformation matrix to a Vec<Vertex> and return the transformed points
        /// parallelized version of the method.
        pub fn transform_points_4x3_par(
            &self,
            points: &Vec<Vertex>,
            transformation_matrix: [[f64; 3]; 4],
        ) -> Vec<Vertex> {
            points
                .par_iter()
                .map(|point| self.multiply_matrix_vector_4x3(transformation_matrix, *point))
                .collect()
        }

        /// Apply a transformation matrix to a mutable Vec<Vertex>.
        pub fn transform_points_mut_4x3(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 3]; 4],
        ) {
            points.iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector_4x3(transformation_matrix, *point);
            });
        }

        /// Apply a transformation matrix to a mutable Vec<Vertex> parallelized version.
        /// parallelized version of the method.
        pub fn transform_points_mut_4x3_par(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 3]; 4],
        ) {
            points.par_iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector_4x3(transformation_matrix, *point);
            });
        }
        pub fn rotation_matrix_from_angles_4x3(
            &self,
            x_angle: f64,
            y_angle: f64,
            z_angle: f64,
        ) -> [[f64; 3]; 4] {
            // Convert angles from degrees to radians
            let x_rad = x_angle.to_radians();
            let y_rad = y_angle.to_radians();
            let z_rad = z_angle.to_radians();

            // Rotation matrix around the X-axis
            let rotation_x = [
                [1.0, 0.0, 0.0],
                [0.0, x_rad.cos(), -x_rad.sin()],
                [0.0, x_rad.sin(), x_rad.cos()],
            ];

            // Rotation matrix around the Y-axis
            let rotation_y = [
                [y_rad.cos(), 0.0, y_rad.sin()],
                [0.0, 1.0, 0.0],
                [-y_rad.sin(), 0.0, y_rad.cos()],
            ];

            // Rotation matrix around the Z-axis
            let rotation_z = [
                [z_rad.cos(), -z_rad.sin(), 0.0],
                [z_rad.sin(), z_rad.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ];

            // Combine the rotation matrices: R = Rz * Ry * Rx
            let rotation_xy = self.multiply_matrices_3x3(rotation_y, rotation_x);
            let rotation_xyz = self.multiply_matrices_3x3(rotation_z, rotation_xy);

            // Embed the 3x3 matrix into a 4x3 matrix
            [
                [rotation_xyz[0][0], rotation_xyz[0][1], rotation_xyz[0][2]],
                [rotation_xyz[1][0], rotation_xyz[1][1], rotation_xyz[1][2]],
                [rotation_xyz[2][0], rotation_xyz[2][1], rotation_xyz[2][2]],
                [0.0, 0.0, 0.0], // Fourth row is zero for a 4x3 matrix
            ]
        }

        /// Utility function: Multiply two 3x3 matrices
        fn multiply_matrices_3x3(&self, a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
            let mut result = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
                }
            }
            result
        }

        /// Helper function: Multiply a 4x3 matrix with a Vertex as translation vector.
        fn multiply_matrix_vector_4x3(&self, matrix: [[f64; 3]; 4], point: Vertex) -> Vertex {
            Vertex::new(
                matrix[0][0] * point.x
                    + matrix[0][1] * point.y
                    + matrix[0][2] * point.z
                    + matrix[3][0],
                matrix[1][0] * point.x
                    + matrix[1][1] * point.y
                    + matrix[1][2] * point.z
                    + matrix[3][1],
                matrix[2][0] * point.x
                    + matrix[2][1] * point.y
                    + matrix[2][2] * point.z
                    + matrix[3][2],
            )
        }

        /// Generate a panning transformation matrix
        pub fn get_pan_matrix_4x3(&self, right_amount: f64, up_amount: f64) -> [[f64; 3]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vector3d::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .unitize_b();

            // Step 2: Compute the right direction vector (cross product of forward and up)
            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();

            // Step 3: Compute the true up direction (orthogonalized)
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Step 4: Compute the translation
            let pan_translation =
                right.to_vertex() * (-right_amount) + (up.to_vertex() * up_amount);

            // Step 5: Construct the 4x3 pan matrix
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [pan_translation.x, pan_translation.y, pan_translation.z],
            ]
        }

        /// Apply panning to a set of 3D points
        pub fn pan_points_4x3(
            &self,
            points: &Vec<Vertex>,
            right_amount: f64,
            up_amount: f64,
        ) -> Vec<Vertex> {
            let pan_matrix = self.get_pan_matrix_4x3(right_amount, up_amount);
            points
                .iter()
                .map(|point| self.multiply_matrix_vector_4x3(pan_matrix, *point))
                .collect()
        }

        /// Apply panning to a set of mutable 3D points
        pub fn pan_points_mut_4x3(
            &self,
            points: &mut Vec<Vertex>,
            right_amount: f64,
            up_amount: f64,
        ) {
            let pan_matrix = self.get_pan_matrix_4x3(right_amount, up_amount);
            points.iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector_4x3(pan_matrix, *point);
            });
        }
        /// Combine multiple transformation matrices into one
        /// They have to be called from a stack vector using the macro `vec!` for that.
        pub fn combine_matrices_4x3(matrices: Vec<[[f64; 3]; 4]>) -> [[f64; 3]; 4] {
            // Start with the identity matrix
            let mut result = [
                [1.0, 0.0, 0.0], // First row
                [0.0, 1.0, 0.0], // Second row
                [0.0, 0.0, 1.0], // Third row
                [0.0, 0.0, 0.0], // Translation row
            ];

            // Multiply all matrices together in sequence
            for matrix in matrices {
                result = Self::multiply_matrices_4x3(result, matrix);
            }

            result
        }
        /// Helper function to multiply two 4x3 matrices
        pub fn multiply_matrices_4x3(a: [[f64; 3]; 4], b: [[f64; 3]; 4]) -> [[f64; 3]; 4] {
            let mut result = [[0.0; 3]; 4];

            // Multiply the rotation part (top 3x3 part of the matrices)
            for i in 0..3 {
                for j in 0..3 {
                    result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
                }
            }

            // Multiply the translation part
            for i in 0..3 {
                result[3][i] = a[3][0] * b[0][i] + a[3][1] * b[1][i] + a[3][2] * b[2][i] + b[3][i];
            }

            result
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
}

pub mod visualization_v4 {
    use super::rendering_object::Vertex;
    use crate::rust3d::geometry::Vector3d;
    use rayon::prelude::*;
    /*
    - Camera Space mapping:
    | Right.x   Right.y   Right.z   Tx |
    | Up.x      Up.y      Up.z      Ty |
    | Forward.x Forward.y Forward.z Tz |
    | 0.0        0.0      0.0        1 |
    */
    pub struct Camera {
        // for tracking the delta camera position.
        pub initial_position: Vertex,
        pub initial_target: Vertex,
        pub initial_right: Vertex,
        pub initial_forward: Vertex,
        pub initial_up: Vertex,
        // for caching the camera position.
        pub position: Vertex,
        pub target: Vertex,
        pub up: Vector3d,
        pub cam_up: Vector3d,
        pub cam_right: Vector3d,
        pub cam_forward: Vector3d,
        // for projection system computation.
        pub fov: f64,
        pub width: f64,
        pub height: f64,
        pub near: f64,
        pub far: f64,
        pub view_matrix: [[f64; 4]; 4],
        pub projection_matrix: [[f64; 4]; 4],
    }
    impl Camera {
        pub fn new(width: f64, height: f64, fov: f64, near: f64, far: f64) -> Self {
            // Default system projection settings
            // User setting will be just an offset of that initial setting.
            // User setting will be implemented later...
            //////////////////////////////////////////////////////////////////
            // Sytem projection initial settings:
            // will be offseted by user setting and (scaled up/or down)
            // when fully implemented.
            let position = Vertex::new(0.0, -1.0, 0.3);
            let target = Vertex::new(0.0, 0.0, 0.0);
            let up_system = Vector3d::new(0.0, 0.0, 1.0);
            //////////////////////////////////////////////////////////////////
            // Compute inital components for mapping the world sytem orientation 
            // in the right orientation from intial setting inputs reference (above).
            let right_direction = Vertex::new(1.0, 0.0, 0.0);
            let forward_direction = (target - position).normalize();
            let up_direction = -forward_direction.cross(&right_direction).normalize();
            //////////////////////////////////////////////////////////////////
            let mut camera = Self {
                initial_position: position,
                initial_target: target,
                initial_right: right_direction,
                initial_forward: forward_direction,
                initial_up: up_direction,
                position,
                target,
                up: up_system,
                cam_up: up_direction.to_vector3d(),
                cam_right: right_direction.to_vector3d(),
                cam_forward: forward_direction.to_vector3d(),
                fov,
                width,
                height,
                near,
                far,
                view_matrix: [[0.0; 4]; 4],
                projection_matrix: [[0.0; 4]; 4],
            };
            // Reverse z for inverting the projection of the camera view_matrix
            // point of view from universe (opposit).
            camera.position.z = -camera.position.z;
            // Precompute the two matrix (camera space & projection).
            camera.view_matrix = camera.compute_view_matrix();
            camera.projection_matrix = camera.compute_projection_matrix();
            // Remap back, the camera z from view projection matrix to match
            // the world coordinate system orientation.
            camera.position.z = -camera.position.z;
            camera
        }

        /// Compute the view matrix space coordinates
        /// (representing the camera transformation)
        fn compute_view_matrix(&self) -> [[f64; 4]; 4] {
            let forward = Vertex::new(
                self.target.x - self.position.x,
                self.target.y - self.position.y,
                self.target.z - self.position.z,
            )
            .normalize()
            .reverse();
            let mut right = forward.cross(&self.up.to_vertex()).normalize();
            // map the (y,z) polarity to match the world space space
            // from intial projection.
            right.y = -right.y;
            right.z = -right.z;
            let up = right.cross(&forward).normalize();
            let translation = Vertex::new(-self.position.x, -self.position.y, -self.position.z);
            [
                [right.x, up.x, forward.x, 0.0],
                [right.y, up.y, forward.y, 0.0],
                [right.z, up.z, forward.z, 0.0],
                [
                    (right.x * translation.x)
                        + (right.y * translation.y)
                        + (right.z * translation.z),
                    (up.x * translation.x) + (up.y * translation.y) + (up.z * translation.z),
                    // notes: (old state) right.z * translation.z (upper line last op component)
                    (forward.x * translation.x)
                        + (forward.y * translation.y)
                        + (forward.z * translation.z),
                    1.0,
                ],
            ]
        }

        /// Compute the projection matrix.
        /// ( map 3d coordinates on 2d screen space )
        fn compute_projection_matrix(&self) -> [[f64; 4]; 4] {
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

        /// Project a set of Vertex on 2d screen space.
        /// # Returns
        /// a vector of tuple (x,y) (usize,usize)
        pub fn project_a_list_of_points(&self, points: &Vec<Vertex>) -> Vec<(usize, usize)> {
            points
                .par_iter() // Use parallel iterator
                .filter_map(|point| self.project_without_depth(point)) // Apply the project method in parallel
                .collect() // Collect results into a Vec
        }

        /// Project a set of Vertex on 2d screen space with
        /// an embeded depth value stored by points.
        /// # Returns
        /// a vector of tuple (x,y,depth) (usize,usize,f64)
        pub fn project_a_list_of_points_with_depth(
            &self,
            points: &Vec<Vertex>,
        ) -> Vec<(usize, usize, f64)> {
            points
                .par_iter() // Use parallel iterator
                .filter_map(|point| self.project_with_depth(point)) // Apply the project method in parallel
                .collect() // Collect results into a Vec
        }

        /// Project a vertex in a 2d camera space projection.
        pub fn project_without_depth(&self, point: &Vertex) -> Option<(usize, usize)> {
            // Use precomputed matrices
            let camera_space_point = self.multiply_matrix_vector(&self.view_matrix, point);

            let projected_point =
                self.multiply_matrix_vector(&self.projection_matrix, &camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.x / projected_point.z;
            let y = projected_point.y / projected_point.z;

            // Map the coordinates from [-1, 1] to screen space
            let screen_x = ((x + 1.0) * 0.5 * self.width) as isize;
            let screen_y = ((1.0 - y) * 0.5 * self.height) as isize;

            if screen_x < 0
                || screen_x >= self.width as isize
                || screen_y < 0
                || screen_y >= self.height as isize
            {
                None // Point is out of screen bounds
            } else {
                Some((screen_x as usize, screen_y as usize))
            }
        }

        /// Project a vertex in a 2d camera space projection with embeded depth value.
        pub fn project_with_depth(&self, point: &Vertex) -> Option<(usize, usize, f64)> {
            // Use precomputed matrices
            let camera_space_point = self.multiply_matrix_vector(&self.view_matrix, point);

            // Extract depth in camera space (before projection)
            let depth_in_camera_space = camera_space_point.z;

            let projected_point =
                self.multiply_matrix_vector(&self.projection_matrix, &camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.x / projected_point.z;
            let y = projected_point.y / projected_point.z;

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

            Some((screen_x as usize, screen_y as usize, depth_in_camera_space))
        }

        #[inline(always)]
        /// Project a single vertex in the 2d camera space projection
        /// where the result projection may be out of the screen limit.
        /// (usefull for drawing  lines axis).
        pub fn project_maybe_outside(&self, point: &Vertex) -> (f64, f64) {
            // Use precomputed matrices
            let camera_space_point = self.multiply_matrix_vector(&self.view_matrix, point);

            let projected_point =
                self.multiply_matrix_vector(&self.projection_matrix, &camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.x / projected_point.z;
            let y = projected_point.y / projected_point.z;

            // Map the coordinates from [-1, 1] to screen space
            let screen_x = ((x + 1.0) * 0.5 * self.width) as isize;
            let screen_y = ((1.0 - y) * 0.5 * self.height) as isize;

            (screen_x as f64, screen_y as f64)
        }

        ///need update...do not use
        /// Generate a panning transformation matrix
        /// `dx` and `dy` are the offsets in world space along the right and up directions.
        pub fn transform_camera_matrix_pan(&self, dx: f64, dy: f64) -> [[f64; 4]; 4] {
            // Calculate the right and up vectors based on the camera's orientation
            let forward = Vector3d::new(
                self.target.x - self.position.x,
                self.target.y - self.position.y,
                self.target.z - self.position.z,
            )
            .unitize_b();

            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Translation in the right and up directions
            let translation = Vertex::new(
                dx * right.get_X() + dy * up.get_X(),
                dx * right.get_Y() + dy * up.get_Y(),
                dx * right.get_Z() + dy * up.get_Z(),
            );

            // Construct the transformation matrix
            [
                [1.0, 0.0, 0.0, translation.x],
                [0.0, 1.0, 0.0, translation.y],
                [0.0, 0.0, 1.0, translation.z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }

        ///need update...do not use
        /// Generate a transformation matrix for the camera's movement and rotation
        pub fn transform_camera_matrix_position_4x4(
            &self,
            forward_amount: f64,
            yaw_angle: f64,
            pitch_angle: f64,
        ) -> [[f64; 4]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vertex::new(
                self.target.x - self.position.x,
                self.target.y - self.position.y,
                self.target.z - self.position.z,
            )
            .normalize();

            // Step 2: Compute the translation vector for the camera movement
            let translation = forward * (-forward_amount);

            // Step 3: Compute rotation matrices
            // Yaw (rotation around the Y-axis)
            let yaw_cos = yaw_angle.to_radians().cos();
            let yaw_sin = yaw_angle.to_radians().sin();
            let yaw_matrix = [
                [yaw_cos, 0.0, yaw_sin, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-yaw_sin, 0.0, yaw_cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Pitch (rotation around the X-axis)
            let pitch_cos = pitch_angle.to_radians().cos();
            let pitch_sin = pitch_angle.to_radians().sin();
            let pitch_matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, pitch_cos, -pitch_sin, 0.0],
                [0.0, pitch_sin, pitch_cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Step 4: Combine yaw and pitch into a single rotation matrix
            let rotation_matrix = self.multiply_matrices(&yaw_matrix, &pitch_matrix);

            // Step 5: Create the translation matrix
            let translation_matrix = [
                [1.0, 0.0, 0.0, translation.x],
                [0.0, 1.0, 0.0, translation.y],
                [0.0, 0.0, 1.0, translation.z],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Step 6: Combine rotation and translation matrices into a single transformation matrix
            self.multiply_matrices(&rotation_matrix, &translation_matrix)
        }

        #[inline(always)]
        /// Utility function to apply a translation vector to 4x4 matrix.
        pub fn multiply_matrix_vector(&self, matrix: &[[f64; 4]; 4], v: &Vertex) -> Vertex {
            Vertex::new(
                matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z + matrix[0][3],
                matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z + matrix[1][3],
                matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z + matrix[2][3],
            )
        }

        /// Utility Combine multiple transformation matrices into one
        /// they have to be call from a stack vector use macro vec! for that.
        pub fn combine_matrices(&self, matrices: Vec<[[f64; 4]; 4]>) -> [[f64; 4]; 4] {
            // Start with the identity matrix
            let mut result = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Multiply all matrices together in sequence
            for matrix in matrices {
                result = self.multiply_matrices(&result, &matrix);
            }

            result
        }

        /// Utility function: Multiply two 4x4 matrices.
        fn multiply_matrices(&self, a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
            let mut result = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    result[i][j] = a[i][0] * b[0][j]
                        + a[i][1] * b[1][j]
                        + a[i][2] * b[2][j]
                        + a[i][3] * b[3][j];
                }
            }
            result
        }

        #[inline(always)]
        /// Extrat the Camera Direction (forward) of the view matrix.
        pub fn get_camera_direction(&self) -> Vertex {
            Vertex::new(
                self.view_matrix[2][0],
                self.view_matrix[2][1],
                self.view_matrix[2][2],
            )
        }

        #[inline(always)]
        /// Extract the camera Up direction from the view matrix.
        pub fn get_camera_up(&self) -> Vertex {
            Vertex::new(
                -self.view_matrix[1][0],
                -self.view_matrix[1][1],
                -self.view_matrix[1][2],
            )
        }

        #[inline(always)]
        /// Extract the camera Right direction from the view matrix.
        pub fn get_camera_right(&self) -> Vertex {
            Vertex::new(
                -self.view_matrix[0][0],
                -self.view_matrix[0][1],
                self.view_matrix[0][2],
            )
        }
        ////////////////////////////////////////////////////////////////////////
        // scratch code do not use (below that line)
        ////////////////////////////////////////////////////////////////////////
        /*
        #[inline(always)]
        pub fn get_camera_target(&self) -> Vertex {
            self.get_camera_position()
                + (self.get_camera_direction().normalize() * (self.target - self.position).Length())
        }
        */

        #[inline(always)]
        fn get_camera_position(&self) -> Vertex {
            let right = Vertex::new(
                self.view_matrix[0][0],
                self.view_matrix[1][0],
                self.view_matrix[2][0],
            );
            let up = Vertex::new(
                self.view_matrix[0][1],
                self.view_matrix[1][1],
                self.view_matrix[2][1],
            );
            let forward = Vertex::new(
                self.view_matrix[0][2],
                self.view_matrix[1][2],
                self.view_matrix[2][2],
            );
            /*
            let translation = Vertex::new(
                self.view_matrix[0][3],
                self.view_matrix[1][3],
                self.view_matrix[2][3],
            );
            */
            let translation = Vertex::new(1.0, 1.0, 1.0);
            // Camera origin is the negation of the transformed translation
            // The camera's world position is the inverse of the rotation times translation
            Vertex::new(
                right.x * translation.x + right.y * translation.y + right.z * translation.z,
                up.x * translation.x + up.y * translation.y + up.z * translation.z,
                forward.x * translation.x + forward.y * translation.y + forward.z * translation.z,
            )
        }

        fn extract_camera_position(&self) -> (f64, f64, f64) {
            let right = (
                self.view_matrix[0][0],
                self.view_matrix[1][0],
                self.view_matrix[2][0],
            );
            let up = (
                self.view_matrix[0][1],
                self.view_matrix[1][1],
                self.view_matrix[2][1],
            );
            let forward = (
                self.view_matrix[0][2],
                self.view_matrix[1][2],
                self.view_matrix[2][2],
            );
            let translation = (
                self.view_matrix[3][0],
                self.view_matrix[3][1],
                self.view_matrix[3][2],
            );
            // Compute world position using the inverse transform
            let x = -(right.0 * translation.0 + right.1 * translation.1 + right.2 * translation.2);
            let y = -(up.0 * translation.0 + up.1 * translation.1 + up.2 * translation.2);
            let z = -(forward.0 * translation.0
                + forward.1 * translation.1
                + forward.2 * translation.2);
            (-x, -y, -z)
        }

        fn extract_camera_target(&self) -> (f64, f64, f64) {
            let (px, py, pz) = self.extract_camera_position();
            // Forward vector (negative Z-axis in view space)
            let forward = (
                -self.view_matrix[0][2],
                -self.view_matrix[1][2],
                -self.view_matrix[2][2],
            );
            let target_x = px + forward.0;
            let target_y = py + forward.1;
            let target_z = pz + forward.2;
            (target_x, target_y, target_z)
        }

        /// Compute the inverse of a 4x4 matrix
        fn inverse_matrix(&self) -> [[f64; 4]; 4] {
            let mut inv = [[0.0; 4]; 4];
            let m = self.view_matrix;

            // Compute inverse using Gaussian elimination or optimized inverse function
            // Extract translation component from the inverse matrix
            inv[0][3] = -(m[0][0] * m[0][3] + m[1][0] * m[1][3] + m[2][0] * m[2][3]);
            inv[1][3] = -(m[0][1] * m[0][3] + m[1][1] * m[1][3] + m[2][1] * m[2][3]);
            inv[2][3] = -(m[0][2] * m[0][3] + m[1][2] * m[1][3] + m[2][2] * m[2][3]);
            inv
        }
    }
}

/*
 *   WORKINPROGRES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *   New mesh prototype with shared vertices.
 *   like this moving points location will also move
 *   faces.
 *   - triangles contain only index id of the vertex.
 *   - and normal vectors
 */
pub mod rendering_object {
    use crate::rust3d::geometry::{CPlane, Point3d, Vector3d};
    use crate::rust3d::intersection::clip_line;
    use crate::rust3d::{self, transformation};
    use dashmap::DashMap;
    use iter::{IntoParallelRefMutIterator, ParallelIterator};
    use rayon::prelude::*; // For parallel processing
    use rayon::*;
    use std::collections::HashMap;
    use std::fs::File;
    use std::hash::{Hash, Hasher};
    use std::io;
    use std::io::{BufRead, BufReader};
    use std::io::{BufWriter, Write};
    use std::marker::Copy;
    use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};
    use std::sync::{Arc, Mutex};
    /// A 3D vertex or point.
    #[derive(Debug, Clone, Copy, PartialOrd)]
    pub struct Vertex {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }
    impl Vertex {
        /// Creates a new Vertex.
        pub fn new(x: f64, y: f64, z: f64) -> Self {
            Vertex { x, y, z }
        }
        /// Computes the cross product of two vectors.
        pub fn cross(self, other: &Vertex) -> Vertex {
            Vertex {
                x: self.y * other.z - self.z * other.y,
                y: self.z * other.x - self.x * other.z,
                z: self.x * other.y - self.y * other.x,
            }
        }
        /// Computes the magnitude of the vector.
        pub fn magnitude(self) -> f64 {
            (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
        }
        /// Normalizes the vector.
        pub fn normalize(self) -> Self {
            let mag = self.magnitude();
            Vertex {
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            }
        }
        #[inline(always)]
        /// Reverse the vertex as a vector.
        pub fn reverse(&self) -> Self {
            Self {
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        }
        /// Test if a point is inside a given mesh.
        pub fn is_inside_a_mesh(&self, mesh_to_test: &Mesh) -> bool {
            let direction = Vertex::new(1.0, 0.0, 0.0);
            let ray = Ray::new(*self, direction);
            let mut intersection_count = 0;

            // Use a HashSet to store truncated distances as integers for hashing
            use std::collections::HashSet;
            let mut distances = HashSet::new();

            for triangle in mesh_to_test.triangles.iter() {
                if let Some(t) = triangle.intersect(&ray, &mesh_to_test.vertices) {
                    // Scale and truncate to an integer to make it hashable
                    let scaled_t = (t * 10000.0).trunc() as i64;

                    // Only count the intersection if it's unique
                    if distances.insert(scaled_t) {
                        intersection_count += 1;
                    }
                }
            }
            // Odd intersection count means the point is inside the mesh
            (intersection_count % 2) != 0
        }
        #[inline(always)]
        pub fn to_point3d(&self) -> Point3d {
            Point3d::new(self.x, self.y, self.z)
        }
        #[inline(always)]
        pub fn to_vector3d(&self) -> Vector3d {
            Vector3d::new(self.x, self.y, self.z)
        }

        /// Computes the dot product of two vertices
        pub fn dot(&self, other: &Vertex) -> f64 {
            self.x * other.x + self.y * other.y + self.z * other.z
        }

        #[inline(always)]
        /// Stabilize double precision number to a digit scale.
        ///  1e3 will round to 0.001
        ///  1e6 will rount to 0.000001
        pub fn clean_up_digits(&mut self, precision: f64) {
            self.x = self.x.trunc() + (self.x.fract() * precision).round() / precision;
            self.y = self.y.trunc() + (self.y.fract() * precision).round() / precision;
            self.z = self.z.trunc() + (self.z.fract() * precision).round() / precision;
        }
    }
    impl Sub for Vertex {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            Vertex::new(self.x - other.x, self.y - other.y, self.z - other.z)
        }
    }
    impl Add for Vertex {
        type Output = Self; // Specify the result type of the addition
        fn add(self, other: Self) -> Self {
            Vertex::new(self.x + other.x, self.y + other.y, self.z + other.z)
        }
    }
    impl Div<f64> for Vertex {
        type Output = Self; // Specify the result type of the addition
        fn div(self, scalar: f64) -> Self {
            Vertex::new(self.x / scalar, self.y / scalar, self.z / scalar)
        }
    }
    //dot product.
    impl Mul for Vertex {
        type Output = f64;
        fn mul(self, other: Vertex) -> f64 {
            self.x * other.x + self.y * other.y + self.z * other.z
        }
    }
    impl PartialEq for Vertex {
        fn eq(&self, other: &Self) -> bool {
            self.x == other.x && self.y == other.y && self.z == other.z
        }
    }
    impl Eq for Vertex {}
    use std::fmt;
    impl fmt::Display for Vertex {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "Vertex(x: {:.3}, y: {:.3}, z: {:.3})",
                self.x, self.y, self.z
            )
        }
    }
    impl AddAssign<Vertex> for Vertex {
        fn add_assign(&mut self, other: Vertex) {
            self.x = self.x + other.x;
            self.y = self.y + other.y;
            self.z = self.z + other.z;
        }
    }
    impl MulAssign<f64> for Vertex {
        fn mul_assign(&mut self, scalar: f64) {
            self.x *= scalar;
            self.y *= scalar;
            self.z *= scalar;
        }
    }
    impl Hash for Vertex {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.x.to_bits().hash(state);
            self.y.to_bits().hash(state);
            self.z.to_bits().hash(state);
        }
    }
    impl Mul<f64> for Vertex {
        type Output = Vertex;
        fn mul(self, scalar: f64) -> Self {
            let v_x = self.x * scalar;
            let v_y = self.y * scalar;
            let v_z = self.z * scalar;
            Vertex::new(v_x, v_y, v_z)
        }
    }
    impl Neg for Vertex {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Self {
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        }
    }
    /// A triangle that references vertices in the shared vertex pool.
    #[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
    pub struct Triangle {
        pub vertex_indices: [usize; 3], // Indices into the vertex pool
        pub normal: Vertex,             // The normal vector for the triangle
        pub id: Option<u64>,
    }
    impl Triangle {
        /// Set a triangle Id
        pub fn set_id(&mut self, id: u64) {
            self.id = Some(id);
        }
        /// provide a copy of the triangle id.
        pub fn get_id(&self) -> Option<u64> {
            self.id
        }
        /// Compute the area of the triangle.
        pub fn get_triangle_area(&self, vertices: &Vec<Vertex>) -> f64 {
            let va = vertices[self.vertex_indices[1]] - vertices[self.vertex_indices[0]];
            let vb = vertices[self.vertex_indices[2]] - vertices[self.vertex_indices[0]];
            va.cross(&vb).magnitude() / 2.0
        }
        /// Compute the signed volume of the tetrahedron formed by the triangle and the origin
        pub fn signed_volume(&self, vertices: &Vec<Vertex>) -> f64 {
            let v0 = &vertices[self.vertex_indices[0]];
            let v1 = &vertices[self.vertex_indices[1]];
            let v2 = &vertices[self.vertex_indices[2]];
            ((*v0).x * ((*v1).y * (*v2).z - (*v1).z * (*v2).y)
                - (*v0).y * ((*v1).x * (*v2).z - (*v1).z * (*v2).x)
                + (*v0).z * ((*v1).x * (*v2).y - (*v1).y * (*v2).x))
                / 6.0
        }
        /// Creates a new Triangle and computes its normal.
        pub fn new(vertices: &[Vertex], vertex_indices: [usize; 3]) -> Self {
            let v0 = vertices[vertex_indices[0]];
            let v1 = vertices[vertex_indices[1]];
            let v2 = vertices[vertex_indices[2]];
            // Compute the normal using the cross product of two edges
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = edge1.cross(&edge2).normalize();
            Self {
                vertex_indices,
                normal,
                id: None,
            }
        }
        /// Recomputes the normal for this triangle based on the current vertices.
        pub fn recompute_normal(&mut self, vertices: &[Vertex]) {
            let v0 = vertices[self.vertex_indices[0]];
            let v1 = vertices[self.vertex_indices[1]];
            let v2 = vertices[self.vertex_indices[2]];
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            self.normal = edge1.cross(&edge2).normalize();
        }

        // Give back edges components.
        pub fn edges(&self, vertices: &Vec<Vertex>) -> [(Vertex, Vertex); 3] {
            let ref v0 = vertices[self.vertex_indices[0]];
            let ref v1 = vertices[self.vertex_indices[1]];
            let ref v2 = vertices[self.vertex_indices[2]];
            [(*v0, *v1), (*v1, *v2), (*v2, *v0)]
        }
        pub fn with_indices_and_normal(v0: usize, v1: usize, v2: usize, normal: Vertex) -> Self {
            Self {
                vertex_indices: [v0, v1, v2],
                normal,
                id: None,
            }
        }
        pub fn with_indices(v0: usize, v1: usize, v2: usize, vertices: &[Vertex]) -> Self {
            let p0 = vertices[v0];
            let p1 = vertices[v1];
            let p2 = vertices[v2];
            let normal = (p1 - p0).cross(&(p2 - p0)).normalize();
            Self {
                vertex_indices: [v0, v1, v2],
                normal,
                id: None,
            }
        }

        /// Ray-triangle intersection using the Möller–Trumbore algorithm.
        /// Returns `Some(t)` where `t` is the distance along the ray, or `None` if no intersection occurs.
        pub fn intersect(
            &self,
            ray: &Ray,
            vertices: &[Vertex], // Shared vertices from the Mesh
        ) -> Option<f64> {
            // Retrieve vertices of the triangle using their indices
            let v0 = vertices[self.vertex_indices[0]];
            let v1 = vertices[self.vertex_indices[1]];
            let v2 = vertices[self.vertex_indices[2]];

            // Edge vectors of the triangle
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;

            // Compute determinant
            let h = ray.direction.cross(&edge2);
            let det = edge1.dot(&h);

            // If the determinant is near zero, the ray lies in the triangle's plane
            if det.abs() < 1e-8 {
                return None;
            }

            let inv_det = 1.0 / det;

            // Calculate distance from v0 to ray origin
            let s = ray.origin - v0;

            // Calculate u parameter and test bounds
            let u = s.dot(&h) * inv_det;
            if u < 0.0 || u > 1.0 {
                return None;
            }

            // Calculate v parameter and test bounds
            let q = s.cross(&edge1);
            let v = ray.direction.dot(&q) * inv_det;
            if v < 0.0 || u + v > 1.0 {
                return None;
            }

            // Calculate t to find the intersection point
            let t = edge2.dot(&q) * inv_det;

            // If t is positive, the ray intersects the triangle
            if t > 1e-8 {
                Some(t)
            } else {
                None
            }
        }

        /// Helper to retrieve vertices of the triangle from the mesh
        pub fn get_vertices(&self, vertices: &[Vertex]) -> [Vertex; 3] {
            [
                vertices[self.vertex_indices[0]],
                vertices[self.vertex_indices[1]],
                vertices[self.vertex_indices[2]],
            ]
        }

        // Compute the centroid of the triangle
        pub fn center(&self, vertices: &Vec<Vertex>) -> [f64; 3] {
            let v0 = vertices[self.vertex_indices[0]];
            let v1 = vertices[self.vertex_indices[1]];
            let v2 = vertices[self.vertex_indices[2]];
            let centroid = Vertex {
                x: (v0.x + v1.x + v2.x) / 3.0,
                y: (v0.y + v1.y + v2.y) / 3.0,
                z: (v0.z + v1.z + v2.z) / 3.0,
            };
            [centroid.x, centroid.y, centroid.z]
        }
        // Compute the centroid of the triangle
        pub fn center_to_vertex(&self, vertices: &Vec<Vertex>) -> Vertex {
            let v0 = vertices[self.vertex_indices[0]];
            let v1 = vertices[self.vertex_indices[1]];
            let v2 = vertices[self.vertex_indices[2]];
            let centroid = Vertex {
                x: (v0.x + v1.x + v2.x) / 3.0,
                y: (v0.y + v1.y + v2.y) / 3.0,
                z: (v0.z + v1.z + v2.z) / 3.0,
            };
            Vertex::new(centroid.x, centroid.y, centroid.z)
        }
        /// Tests for intersection between a ray and this triangle.
        /// Returns `Some(t)` where `t` is the distance along the ray to the intersection,
        /// or `None` if there's no intersection.
        pub fn intersect_vb(&self, ray: &Ray, vertices: &[Vertex]) -> Option<f64> {
            let v0 = vertices[self.vertex_indices[0]];
            let v1 = vertices[self.vertex_indices[1]];
            let v2 = vertices[self.vertex_indices[2]];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;

            let h = ray.direction.cross(&edge2);
            let a = edge1.dot(&h);

            // Check if the ray is parallel to the triangle (a is near zero)
            if a.abs() < 1e-8 {
                return None;
            }

            let f = 1.0 / a;
            let s = ray.origin - v0;
            let u = f * s.dot(&h);

            // Check if the intersection is outside the triangle
            if u < 0.0 || u > 1.0 {
                return None;
            }

            let q = s.cross(&edge1);
            let v = f * ray.direction.dot(&q);

            // Check if the intersection is outside the triangle
            if v < 0.0 || u + v > 1.0 {
                return None;
            }

            // Compute the distance `t` along the ray
            let t = f * edge2.dot(&q);

            // Check if the intersection is behind the ray origin
            if t > 1e-8 {
                Some(t) // Valid intersection
            } else {
                None // Line intersection but not a ray intersection
            }
        }
    }
    use std::path::Path;
    /// A mesh containing vertices and triangles.
    #[derive(Debug, Clone, PartialEq, PartialOrd)]
    pub struct Mesh {
        pub vertices: Vec<Vertex>,
        pub triangles: Vec<Triangle>,
    }
    impl Mesh {
        /// Creates a new empty mesh.
        pub fn new() -> Self {
            Mesh {
                vertices: Vec::new(),
                triangles: Vec::new(),
            }
        }
        pub fn new_with_data(vertices: Vec<Vertex>, triangles: Vec<Triangle>) -> Self {
            Mesh {
                vertices,
                triangles,
            }
        }
        /// Adds a vertex to the mesh and returns its index.
        pub fn add_vertex(&mut self, vertex: Vertex) -> usize {
            self.vertices.push(vertex);
            self.vertices.len() - 1
        }
        /// Adds a triangle to the mesh by specifying its vertex indices.
        /// Automatically computes and stores the normal for the triangle.
        pub fn add_triangle(&mut self, vertex_indices: [usize; 3]) {
            let triangle = Triangle::new(&self.vertices, vertex_indices);
            self.triangles.push(triangle);
        }
        /// Recomputes the normals for all triangles in the mesh.
        pub fn recompute_normals(&mut self) {
            self.triangles
                .par_iter_mut()
                .for_each(|tri| tri.recompute_normal(&self.vertices));
        }
        /// Give the volume of the mesh in file system base unit.
        /// Based on divergence theorem (also known as Gauss's theorem)
        pub fn compute_volume(&self) -> f64 {
            self.triangles
                .iter()
                .map(|tri| tri.signed_volume(&self.vertices))
                .sum()
        }
        /// make hash map from triangle with id. (slower version from std lib)
        /// Mutex ensure thread safety over threads.
        /// use .get() on option inside value to retrieve triangle from id.
        pub fn make_triangle_hash_map(&self) -> Option<HashMap<u64, &Triangle>> {
            // allocate memory for result.
            let mut tri_hash_map = Mutex::new(HashMap::new());
            self.triangles.par_iter().for_each(|triangle| {
                if let Some(id) = triangle.id {
                    let mut hash = tri_hash_map.lock().unwrap();
                    hash.insert(id, triangle);
                }
            });
            tri_hash_map.lock().ok().map(|hash| hash.clone())
        }

        /// make a dhashmap (external dependency) from triangle with id. (recommended lib from IA)
        /// use .get() on option inside value to retrieve triangle from id.
        pub fn make_triangle_dhash_map(&self) -> Option<HashMap<u64, &Triangle>> {
            // Use DashMap for thread-safe concurrent updates
            let tri_hash_map = DashMap::new();

            // Use par_iter for parallel iteration
            self.triangles.par_iter().for_each(|triangle| {
                if let Some(id) = triangle.id {
                    tri_hash_map.insert(id, triangle);
                }
            });

            // Convert DashMap to HashMap
            let result: HashMap<u64, &Triangle> = tri_hash_map.into_iter().collect();
            Some(result)
        }
        // Import a mesh from an .obj file
        fn import_from_obj<P: AsRef<Path>>(path: P) -> io::Result<Self> {
            let file = File::open(path)?;
            let reader = BufReader::new(file);

            let mut vertices = Vec::new();
            let mut triangles = Vec::new();

            for line in reader.lines() {
                let line = line?;
                let parts: Vec<&str> = line.split_whitespace().collect();

                if parts.is_empty() {
                    continue;
                }

                match parts[0] {
                    "v" => {
                        // Vertex line
                        let x: f64 = parts[1].parse().unwrap();
                        let y: f64 = parts[2].parse().unwrap();
                        let z: f64 = parts[3].parse().unwrap();
                        vertices.push(Vertex::new(x, y, z));
                    }
                    "f" => {
                        // Face line
                        let indices: Vec<usize> = parts[1..]
                            .iter()
                            .map(|p| p.split('/').next().unwrap().parse::<usize>().unwrap() - 1)
                            .collect();
                        if indices.len() == 3 {
                            triangles.push([indices[0], indices[1], indices[2]]);
                        }
                    }
                    _ => {}
                }
            }
            // Build the mesh
            let mut mesh = Mesh::new();
            mesh.vertices = vertices;
            for triangle_indices in triangles {
                mesh.add_triangle(triangle_indices);
            }
            Ok(mesh)
        }

        /// Export the mesh to an .obj file
        fn export_to_obj<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
            let mut file = File::create(path)?;

            // Write vertices
            for vertex in &self.vertices {
                writeln!(file, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
            }

            // Write faces
            for triangle in &self.triangles {
                writeln!(
                    file,
                    "f {} {} {}",
                    triangle.vertex_indices[0] + 1,
                    triangle.vertex_indices[1] + 1,
                    triangle.vertex_indices[2] + 1
                )?;
            }

            Ok(())
        }

        pub fn import_obj_with_normals(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
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
                        normals.push(Vertex::new(x, y, z).normalize());
                    }
                    "f" => {
                        // Face
                        let mut vertex_indices = Vec::new();
                        let mut normal_indices = Vec::new();

                        for part in &parts[1..] {
                            let indices: Vec<&str> = part.split('/').collect();
                            let vertex_idx: usize = indices[0].parse::<usize>()? - 1; // .obj is 1-indexed
                            vertex_indices.push(vertex_idx);

                            // If normals are available
                            if indices.len() > 2 && !indices[2].is_empty() {
                                let normal_idx: usize = indices[2].parse::<usize>()? - 1;
                                normal_indices.push(normal_idx);
                            }
                        }

                        if vertex_indices.len() == 3 {
                            let triangle = if normal_indices.len() == 3 {
                                // Use the first normal for the entire triangle
                                let normal = normals[normal_indices[0]];
                                Triangle::with_indices_and_normal(
                                    vertex_indices[0],
                                    vertex_indices[1],
                                    vertex_indices[2],
                                    normal,
                                )
                            } else {
                                // Compute normal if not provided
                                Triangle::with_indices(
                                    vertex_indices[0],
                                    vertex_indices[1],
                                    vertex_indices[2],
                                    &vertices,
                                )
                            };
                            triangles.push(triangle);
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

        /// Export Mesh to an obj file.
        pub fn export_to_obj_with_normals_fast(
            &self,
            path: &str,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);

            println!("\x1b[2J");
            println!("\x1b[3;0HExporting (optimized) Path:({})", path);

            let vertex_count = self.vertices.len();
            let triangle_count = self.triangles.len();

            // Step 1: Precompute vertex-to-index mapping
            let vertex_index_map: HashMap<usize, usize> = self
                .vertices
                .iter()
                .enumerate()
                .map(|(i, _)| (i, i + 1)) // OBJ indices are 1-based
                .collect();

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
                    let v0_idx = vertex_index_map[&triangle.vertex_indices[0]];
                    let v1_idx = vertex_index_map[&triangle.vertex_indices[1]];
                    let v2_idx = vertex_index_map[&triangle.vertex_indices[2]];
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
        /// Test if a mesh is closed (watertight).
        pub fn is_watertight(&self) -> bool {
            let mut edge_count: HashMap<(Vertex, Vertex), i32> = HashMap::new();
            for triangle in self.triangles.iter() {
                for &(start, end) in &triangle.edges(&self.vertices) {
                    let edge = if start < end {
                        (start, end)
                    } else {
                        (end, start)
                    };
                    let counter = edge_count.entry(edge).or_insert(0);
                    *counter += 1;
                }
            }
            edge_count.values().all(|&count| count == 2)
        }

        /// Test if a mesh is closed (watertight) using parallel processing.
        pub fn is_watertight_par(&self) -> bool {
            let edge_count = Mutex::new(HashMap::new());

            // Process triangles in parallel
            self.triangles.par_iter().for_each(|triangle| {
                for &(start, end) in &triangle.edges(&self.vertices) {
                    let edge = if start < end {
                        (start, end)
                    } else {
                        (end, start)
                    };

                    // Lock the mutex, update the edge count, and immediately release the lock
                    let mut map = edge_count.lock().unwrap();
                    *map.entry(edge).or_insert(0) += 1;
                }
            });

            // Perform the final watertight check
            let final_map = edge_count.lock().unwrap(); // Lock again to check
            final_map.values().all(|&count| count == 2)
        }
        /// Merges another mesh into this mesh while maintaining relationships between vertices and triangles.
        pub fn merge(&mut self, other: &Mesh) {
            // Store the current number of vertices in the mesh.
            let vertex_offset = self.vertices.len();

            // Add the vertices from the other mesh.
            self.vertices.extend_from_slice(&other.vertices);

            // Add the triangles from the other mesh, adjusting their vertex indices.
            for triangle in &other.triangles {
                let adjusted_indices = [
                    triangle.vertex_indices[0] + vertex_offset,
                    triangle.vertex_indices[1] + vertex_offset,
                    triangle.vertex_indices[2] + vertex_offset,
                ];

                // Add the new triangle with adjusted vertex indices.
                self.triangles.push(Triangle {
                    vertex_indices: adjusted_indices,
                    normal: triangle.normal,
                    id: triangle.id, // Preserve the triangle ID if applicable.
                });
            }
        }
        pub fn find_shared_edges(&self) -> HashMap<(usize, usize), Vec<usize>> {
            // Create an hashmap tracking edge by their vertex indices
            // created to a vector of a triangle id.
            let mut map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
            for (triangle_index, triangle) in self.triangles.iter().enumerate() {
                // Make edges by triangles vertex indices.
                let edges = [
                    (triangle.vertex_indices[0], triangle.vertex_indices[1]),
                    (triangle.vertex_indices[1], triangle.vertex_indices[2]),
                    (triangle.vertex_indices[2], triangle.vertex_indices[0]),
                ];
                for (vertex_indice_a, vertex_indice_b) in edges.iter() {
                    // normalize triangles edges order by indices number
                    // for further edges comparisons check.
                    let edge = if (*vertex_indice_a < *vertex_indice_b) {
                        (*vertex_indice_a, *vertex_indice_b)
                    } else {
                        (*vertex_indice_b, *vertex_indice_a)
                    };
                    // Ensure the edge key exist and add a triangle indices as value
                    // the vector is initialized if needed.
                    // ( an array would need to track the current local cursor array for each
                    // edge involving meta data in a struct maybe later...)
                    map.entry(edge).or_insert(Vec::new()).push(triangle_index);
                }
            }
            map
        }
        /// Remove duplicate vertices of the mesh and update triangles
        /// vertex indices efficiently.
        pub fn remove_duplicate_vertices(&mut self) {
            let mut unique_vertices = Vec::new();
            let mut index_map: HashMap<Vertex, usize> = HashMap::new();
            // Create a map tracking vertex key to indices index value.
            for &vertex in self.vertices.iter() {
                if let Some(&new_index) = index_map.get(&vertex) {
                    // Vertex already exists, just update the mapping
                    index_map.insert(vertex, new_index);
                } else {
                    // New unique vertex
                    let new_index = unique_vertices.len();
                    unique_vertices.push(vertex);
                    index_map.insert(vertex, new_index);
                }
            }

            // Update triangle indices to reference the deduplicated vertices
            for triangle in &mut self.triangles {
                for index in &mut triangle.vertex_indices {
                    // map the tracked vertex key ,to the new value index from vertices list.
                    if let Some(&new_index) = index_map.get(&self.vertices[*index]) {
                        *index = new_index;
                    }
                }
            }
            // Replace original vertex list with deduplicated one
            self.vertices = unique_vertices;
        }

        // Extract the edge describing the contour of the mesh.
        pub fn extract_silhouette(
            &self,
            camera_direction: &Vertex,
        ) -> Vec<(Vertex, Vertex, String)> {
            let mut result = Vec::new();
            /*
            println!("Extract silhouette Debug Mode:");
            if self.is_watertight() {
                println!("The mesh is closed.");
            } else {
                println!("The mesh is open.");
            }
            */
            let map = self.find_shared_edges();
            for (edge, triangle_id) in map.iter() {
                // if edge is shared with two triangles.
                if triangle_id.len() == 2 {
                    // Compute visibility from camera direction.
                    let triangle_a_is_visible =
                        if self.triangles[triangle_id[0]].normal.dot(&camera_direction) > 0.0 {
                            true
                        } else {
                            false
                        };
                    let triangle_b_is_visible =
                        if self.triangles[triangle_id[1]].normal.dot(&camera_direction) > 0.0 {
                            true
                        } else {
                            false
                        };
                    if triangle_a_is_visible != triangle_b_is_visible {
                        result.push((self.vertices[edge.0], self.vertices[edge.1],
                        format!(
                            "Shared Edge ({6:?},{7:?}) is a border, bound to triangles index:({0:?}&{1:?}), visibility:\"{2:?},{3:?}\" dot:({4:?},{5:?})",
                            triangle_id[0],
                            triangle_id[1],
                            triangle_a_is_visible,
                            triangle_b_is_visible,
                            self.triangles[triangle_id[0]].normal.dot(&camera_direction),
                            self.triangles[triangle_id[1]].normal.dot(&camera_direction),
                            self.vertices[edge.0],
                            self.vertices[edge.1],
                        )));
                    }
                }
                if triangle_id.len() == 1 {
                    let triangle_a_is_visible =
                        if self.triangles[triangle_id[0]].normal.dot(&camera_direction) < 0.0 {
                            true
                        } else {
                            false
                        };
                    result.push((self.vertices[edge.0], self.vertices[edge.1],
                    format!(
                            "Edge ({3:?},{4:?}) is a border, bound to  triangle {0:?} visibility:\"{1:?}\" dot:{2:?}",
                            triangle_id[0],
                            triangle_a_is_visible,
                            self.triangles[triangle_id[0]].normal.dot(&camera_direction),
                            edge.0,
                            edge.1,
                        )));
                }
            }
            result
        }

        #[inline(always)]
        /// # Returns the center vertex and the normal vector of each faces.
        pub fn extract_faces_normals_vectors(&self) -> Vec<(Vertex, Vertex)> {
            self.triangles
                .iter()
                .map(|triangle| (triangle.center_to_vertex(&self.vertices), triangle.normal))
                .collect()
        }
    }
    ////////////////////////////////////////////////////////////////////////////
    // A temporary object representing a parametric object.
    // (this object will be merge in the implementation
    //                                          of a classic mesh once designed )
    // maybe...
    ////////////////////////////////////////////////////////////////////////////
    #[derive(Clone, Debug)]
    pub struct MeshPlane {
        pub vertices: Vec<Vertex>,
        pub triangles: Vec<Triangle>,
        //pub stitch_logic_side_a: Vec<usize>,
        //pub stitch_logic_side_b: Vec<usize>,
        //pub stitch_logic_side_c: Vec<usize>,
        //pub stitch_logic_side_d: Vec<usize>,
    }

    impl MeshPlane {
        // Construct a plane mesh.
        /// buffer and camera are temporary for the design.
        pub fn new(
            buffer: &mut Vec<u32>,
            screen_width: usize,
            screen_height: usize,
            camera: &super::visualization_v4::Camera,
            matrix: Option<&[[f64; 4]; 3]>,
            construction_plane: &CPlane,
            u_length: f64,
            v_length: f64,
            divide_count_u: usize,
            divide_count_v: usize,
        ) -> Self {
            // Evaluate from (u,v) dimension of the grid.
            let spacing_unit_u = u_length / (divide_count_u as f64);
            let spacing_unit_v = v_length / (divide_count_v as f64);

            // println!("grid size:{:?}", uv_length,);
            // Define memory components.
            let mut grid_points = vec![Vertex::new(0.0, 0.0, 0.0); divide_count_u * divide_count_v];
            let mut pt_u = 0.0;
            let mut pt_v = 0.0;
            // Make a grid of points describing the plane.
            for v in 0..divide_count_v {
                for u in 0..divide_count_u {
                    grid_points[v * divide_count_u + u] = (*construction_plane)
                        .point_on_plane_uv(pt_u, pt_v)
                        .to_vertex();
                    pt_u += spacing_unit_u;
                    if u == divide_count_u - 1 {
                        pt_u = 0.0;
                    }
                }
                pt_v += spacing_unit_v;
            }
            // Apply transformations if needed.
            if let Some(m) = matrix {
                grid_points = transformation::transform_points_4x3(m, &grid_points);
            }
            ////////////////////////////////////////////////////////////////////
            // if the grid cell length is equal to 1.
            let plane_vector_u = if divide_count_u == 1 {
                let mut v_u = construction_plane
                    .point_on_plane_uv(spacing_unit_u, 0.0)
                    .to_vertex();
                if let Some(m) = matrix {
                    v_u = transformation::transform_point_4x3(m, &v_u);
                }
                v_u - grid_points[0 * divide_count_u + 0]
            } else {
                // Compute base vectors u for the actual plane
                grid_points[0 * divide_count_u + 1] - grid_points[0 * divide_count_u + 0]
            };
            ////////////////////////////////////////////
            // if the grid cell length is equal to 1.
            let plane_vector_v = if divide_count_v == 1 {
                let mut v_v = construction_plane
                    .point_on_plane_uv(0.0, spacing_unit_v)
                    .to_vertex();
                if let Some(m) = matrix {
                    v_v = transformation::transform_point_4x3(m, &v_v);
                }
                v_v - grid_points[0 * divide_count_u + 0]
            } else {
                // Compute base vectors u for the actual plane
                grid_points[1 * divide_count_u + 0] - grid_points[0 * divide_count_u + 0]
            };
            ////////////////////////////////////////////////////////////////////

            let mut mesh_plane_result = MeshPlane {
                vertices: Vec::new(),
                triangles: Vec::new(),
                //stitch_logic_side_a: Vec::new(),
                //stitch_logic_side_b: Vec::new(),
                //stitch_logic_side_c: Vec::new(),
                //stitch_logic_side_d: Vec::new(),
            };
            // Build Triangles.
            let mut indices = 0;
            for u in 0..divide_count_u {
                for v in 0..divide_count_v {
                    // quad origin point.
                    let mut vert_a = grid_points[v * divide_count_u + u];
                    // quad point on u direction.
                    let mut vert_b = vert_a + plane_vector_u;
                    // quad point diagonal u+v direction.
                    let mut vert_c =
                        grid_points[v * divide_count_u + u] + (plane_vector_u + plane_vector_v);
                    // quad point on v direction.
                    let mut vert_d = grid_points[v * divide_count_u + u] + (plane_vector_v);
                    // Add first triangle.
                    // index logic 1,2,4
                    vert_a.clean_up_digits(1e6);
                    vert_b.clean_up_digits(1e6);
                    vert_c.clean_up_digits(1e6);
                    vert_d.clean_up_digits(1e6);
                    mesh_plane_result.vertices.push(vert_a);
                    mesh_plane_result.vertices.push(vert_b);
                    mesh_plane_result.vertices.push(vert_d);

                    mesh_plane_result.triangles.push(Triangle::with_indices(
                        indices,
                        indices + 1,
                        indices + 2,
                        &mesh_plane_result.vertices,
                    ));
                    /*
                    // Stitching logic:
                    // |-------------|
                    // |      C      |
                    // |D           B|
                    // |      A      |
                    // |-------------|
                    // First corner logic.
                    if (u == 0) && (v == 0) {
                        mesh_plane_result.stitch_logic_side_a.push(indices);
                        mesh_plane_result.stitch_logic_side_d.push(indices);
                    }
                    // Second Corner logic.
                    if (v == 0) && (u == divide_count_u - 1) {
                        mesh_plane_result.stitch_logic_side_a.push(indices + 1);
                        mesh_plane_result.stitch_logic_side_b.push(indices + 1);
                    }
                    // inner  side A.
                    if (v == 0) && (u != divide_count_u - 1) {
                        mesh_plane_result.stitch_logic_side_a.push(indices + 1);
                    }
                    // inner side D
                    if (u == 0) && (v != divide_count_v - 1) {
                        mesh_plane_result.stitch_logic_side_d.push(indices + 2);
                    }
                    */
                    indices += 3;
                    // Add second triangle.
                    // index logic 2,3,4
                    mesh_plane_result.vertices.push(vert_b);
                    mesh_plane_result.vertices.push(vert_c);
                    mesh_plane_result.vertices.push(vert_d);
                    mesh_plane_result.triangles.push(Triangle::with_indices(
                        indices,
                        indices + 1,
                        indices + 2,
                        &mesh_plane_result.vertices,
                    ));
                    /*
                    // Third Corner logic.
                    if (u == divide_count_u - 1) && (v == divide_count_v - 1) {
                        mesh_plane_result.stitch_logic_side_c.push(indices + 1);
                        mesh_plane_result.stitch_logic_side_b.push(indices + 1);
                    }
                    // Last Corner logic.
                    if (u == 0) && v == (divide_count_v - 1) {
                        mesh_plane_result.stitch_logic_side_c.push(indices + 2);
                        mesh_plane_result.stitch_logic_side_d.push(indices + 2);
                    }
                    // inner side B.
                    if (u == divide_count_u - 1) && (v != divide_count_v - 1) {
                        mesh_plane_result.stitch_logic_side_b.push(indices + 1);
                    }
                    // inner side C
                    if (v == divide_count_v - 1) && (u != divide_count_u - 1) {
                        mesh_plane_result.stitch_logic_side_c.push(indices + 1);
                    }
                    */
                    indices += 3;

                    // Temporary display of the indexing logic./////////////////
                    let p1 = camera.project_maybe_outside(&vert_a);
                    let p2 = camera.project_maybe_outside(&vert_b);
                    let p3 = camera.project_maybe_outside(&vert_c);
                    let p4 = camera.project_maybe_outside(&vert_d);
                    // Graph point projection on screen space (only one triangle for now).
                    if let Some(pt) = clip_line(p1, p2, screen_width, screen_height) {
                        rust3d::draw::draw_aa_line(buffer, screen_width, pt.0, pt.1, 0xff6abd);
                    }
                    if let Some(pt) = clip_line(p2, p4, screen_width, screen_height) {
                        rust3d::draw::draw_aa_line(buffer, screen_width, pt.0, pt.1, 0xff6abd);
                    }
                    if let Some(pt) = clip_line(p4, p1, screen_width, screen_height) {
                        rust3d::draw::draw_aa_line(buffer, screen_width, pt.0, pt.1, 0xff6abd);
                    }
                    if let Some(pt) = clip_line(p3, p4, screen_width, screen_height) {
                        rust3d::draw::draw_aa_line(buffer, screen_width, pt.0, pt.1, 0xff6abd);
                    }
                    if let Some(pt) = clip_line(p2, p3, screen_width, screen_height) {
                        rust3d::draw::draw_aa_line(buffer, screen_width, pt.0, pt.1, 0xff6abd);
                    }
                    ////////////////////////////////////////////////////////////
                }
            }
            mesh_plane_result.remove_duplicate_vertices();
            // return the parametric mesh plane.
            mesh_plane_result
        }

        #[inline(always)]
        /// Flip the triangle normal direction.
        pub fn flip_mesh_plane_normals(&mut self) {
            self.triangles
                .iter_mut()
                .for_each(|triangle| triangle.normal = -triangle.normal);
        }

        #[inline(always)]
        /// Duplicate the MeshBox into a regular mesh
        /// copy.
        pub fn to_mesh(&self) -> Mesh {
            Mesh {
                vertices: self.vertices.clone(),
                triangles: self.triangles.clone(),
            }
        }

        /// Remove duplicate vertices of the mesh and update triangles
        /// vertex indices efficiently.
        pub fn remove_duplicate_vertices(&mut self) {
            let mut unique_vertices = Vec::new();
            let mut index_map: HashMap<Vertex, usize> = HashMap::new();
            // Create a map tracking vertex key to indices index value.
            for &vertex in self.vertices.iter() {
                if let Some(&new_index) = index_map.get(&vertex) {
                    // Vertex already exists, just update the mapping
                    index_map.insert(vertex, new_index);
                } else {
                    // New unique vertex
                    let new_index = unique_vertices.len();
                    unique_vertices.push(vertex);
                    index_map.insert(vertex, new_index);
                }
            }

            // Update triangle indices to reference the deduplicated vertices
            for triangle in &mut self.triangles {
                for index in &mut triangle.vertex_indices {
                    // map the tracked vertex key ,to the new value index from vertices list.
                    if let Some(&new_index) = index_map.get(&self.vertices[*index]) {
                        *index = new_index;
                    }
                }
            }

            // Replace original vertex list with deduplicated one
            self.vertices = unique_vertices;
        }
    }
    ////////////////////////////////////////////////////////////////////////////
    pub struct MeshBox {
        pub vertices: Vec<Vertex>,
        pub triangles: Vec<Triangle>,
    }
    impl MeshBox {
        /// Create a parametric mesh box
        /// this object can be copied into a mesh
        /// via a dedicated method.
        pub fn new(
            buffer: &mut Vec<u32>,
            screen_width: usize,
            screen_height: usize,
            camera: &super::visualization_v4::Camera,
            matrix: Option<&[[f64; 4]; 3]>,
            origin: &Vertex,
            mut direction_u: &mut Vertex,
            mut direction_v: &mut Vertex,
            u_length: f64,
            v_length: f64,
            w_length: f64,
            mut divide_count_u: usize,
            mut divide_count_v: usize,
            mut divide_count_w: usize,
        ) -> Self {
            if divide_count_u == 0 {
                divide_count_u = 1;
            }
            if divide_count_v == 0 {
                divide_count_v = 1;
            }
            if divide_count_w == 0 {
                divide_count_w = 1;
            }
            // Compute the basis vectors direction.
            let direction_w = direction_u.cross(direction_v).normalize();
            *direction_u = direction_u.normalize();
            *direction_v = direction_v.normalize();

            // Compute the 4 anchors points of the cube face
            // (where CPlane origin will be positioned).
            let anchor_sud = origin.to_point3d();
            let anchor_est = (*origin + (*direction_u * u_length)).to_point3d();
            let anchor_north = (*origin + (*direction_v * v_length)).to_point3d();
            let anchor_west = anchor_north;
            let anchor_bottom = anchor_sud;
            let anchor_top = (*origin + (direction_w * w_length)).to_point3d();

            // Create the 6 CPlane(s) of each faces of the cube from user inputs.
            // the logic is 1 Sud , 2 Est, 3 North, 4 West, 5 Bottom and 6 Top.
            let mut faces_list: [MeshPlane; 6] = std::array::from_fn(|_| MeshPlane {
                vertices: Vec::new(),
                triangles: Vec::new(),
            });
            // Cube face 0 (Sud).
            let sud_cplane = CPlane::new_origin_x_aligned_y_oriented(
                &anchor_sud,
                &direction_u.to_point3d(),
                &direction_w.to_point3d(),
            );
            faces_list[0] = MeshPlane::new(
                buffer,
                screen_width,
                screen_height,
                camera,
                matrix,
                &sud_cplane,
                u_length,
                w_length,
                divide_count_u,
                divide_count_w,
            );
            // Cube face 1 (Est).
            let pt_dir_v = anchor_est + direction_v.to_point3d();
            let pt_dir_w = anchor_est + direction_w.to_point3d();
            let est_cplane =
                CPlane::new_origin_x_aligned_y_oriented(&anchor_est, &pt_dir_v, &pt_dir_w);
            faces_list[1] = MeshPlane::new(
                buffer,
                screen_width,
                screen_height,
                camera,
                matrix,
                &est_cplane,
                v_length,
                w_length,
                divide_count_v,
                divide_count_w,
            );
            // Cube face 2 (North).
            let pt_dir_u = anchor_north + direction_u.to_point3d();
            let pt_dir_w = anchor_north + direction_w.to_point3d();
            let north_cplane =
                CPlane::new_origin_x_aligned_y_oriented(&anchor_north, &pt_dir_u, &pt_dir_w);
            faces_list[2] = MeshPlane::new(
                buffer,
                screen_width,
                screen_height,
                camera,
                matrix,
                &north_cplane,
                u_length,
                w_length,
                divide_count_u,
                divide_count_w,
            );
            faces_list[2].flip_mesh_plane_normals();
            // Cube face 3 (west)
            let pt_dir_v = anchor_west + direction_v.reverse().to_point3d();
            let pt_dir_w = anchor_west + direction_w.to_point3d();
            let west_cplane =
                CPlane::new_origin_x_aligned_y_oriented(&anchor_west, &pt_dir_v, &pt_dir_w);
            faces_list[3] = MeshPlane::new(
                buffer,
                screen_width,
                screen_height,
                camera,
                matrix,
                &west_cplane,
                v_length,
                w_length,
                divide_count_v,
                divide_count_w,
            );
            // Cube face 4 (bottom)
            let pt_dir_u = anchor_bottom + direction_u.to_point3d();
            let pt_dir_v = anchor_bottom + direction_v.to_point3d();
            let bottom_cplane =
                CPlane::new_origin_x_aligned_y_oriented(&anchor_bottom, &pt_dir_u, &pt_dir_v);
            faces_list[4] = MeshPlane::new(
                buffer,
                screen_width,
                screen_height,
                camera,
                matrix,
                &bottom_cplane,
                u_length,
                v_length,
                divide_count_u,
                divide_count_v,
            );
            faces_list[4].flip_mesh_plane_normals();
            // Cube face 5 (top)
            let pt_dir_u = anchor_top + direction_u.to_point3d();
            let pt_dir_v = anchor_top + direction_v.to_point3d();
            let top_cplane =
                CPlane::new_origin_x_aligned_y_oriented(&anchor_top, &pt_dir_u, &pt_dir_v);
            faces_list[5] = MeshPlane::new(
                buffer,
                screen_width,
                screen_height,
                camera,
                matrix,
                &top_cplane,
                u_length,
                v_length,
                divide_count_u,
                divide_count_v,
            );

            // Allocate memory for the result.
            let mut result = MeshBox {
                vertices: Vec::new(),
                triangles: Vec::new(),
            };
            // Merge Face into a single list.
            for face in faces_list.iter() {
                // Compute the actual offset indices cursor position.
                let offset = result.vertices.len();
                // Push vertex in memory pool.
                for vertex in face.vertices.iter() {
                    result.vertices.push(*vertex);
                }
                //... Offset the triangle indices from the merged
                //vertex memory pool length.
                for tri in face.triangles.iter() {
                    let vertex_indices = [
                        tri.vertex_indices[0] + offset,
                        tri.vertex_indices[1] + offset,
                        tri.vertex_indices[2] + offset,
                    ];
                    result.triangles.push(Triangle::with_indices_and_normal(
                        tri.vertex_indices[0] + offset,
                        tri.vertex_indices[1] + offset,
                        tri.vertex_indices[2] + offset,
                        tri.normal,
                    ));
                    /*
                    result
                        .triangles
                        .push(Triangle::new(&result.vertices, vertex_indices));
                    */
                }
            }
            // Clean duplicate vertex by tracking vertex equality
            // via hashmap.
            result.remove_duplicate_vertices();
            result
        }

        #[inline(always)]
        /// Duplicate the MeshBox into a regular mesh
        /// copy.
        pub fn to_mesh(&self) -> Mesh {
            Mesh {
                vertices: self.vertices.clone(),
                triangles: self.triangles.clone(),
            }
        }
        /// Remove duplicate vertices of the mesh and update triangles
        /// vertex indices efficiently.
        pub fn remove_duplicate_vertices(&mut self) {
            let mut unique_vertices = Vec::new();
            let mut index_map: HashMap<Vertex, usize> = HashMap::new();
            // Create a map tracking vertex key to indices index value.
            for &vertex in self.vertices.iter() {
                if let Some(&new_index) = index_map.get(&vertex) {
                    // Vertex already exists, just update the mapping
                    index_map.insert(vertex, new_index);
                } else {
                    // New unique vertex
                    let new_index = unique_vertices.len();
                    unique_vertices.push(vertex);
                    index_map.insert(vertex, new_index);
                }
            }

            // Update triangle indices to reference the deduplicated vertices
            for triangle in &mut self.triangles {
                for index in &mut triangle.vertex_indices {
                    // map the tracked vertex key ,to the new value index from vertices list.
                    if let Some(&new_index) = index_map.get(&self.vertices[*index]) {
                        *index = new_index;
                    }
                }
            }

            // Replace original vertex list with deduplicated one
            self.vertices = unique_vertices;
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
    ////////////////////////////////////////////////////////////////////////
    #[derive(Debug, Clone)]
    pub struct AABB {
        pub min: Vertex, // Minimum corner of the bounding box
        pub max: Vertex, // Maximum corner of the bounding box
    }

    impl AABB {
        /// Create a new AABB with infinite bounds.
        pub fn new() -> Self {
            AABB {
                min: Vertex::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
                max: Vertex::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
            }
        }

        /// Expand the bounding box to include a vertex.
        pub fn expand(&mut self, vertex: &Vertex) {
            self.min.x = self.min.x.min(vertex.x);
            self.min.y = self.min.y.min(vertex.y);
            self.min.z = self.min.z.min(vertex.z);
            self.max.x = self.max.x.max(vertex.x);
            self.max.y = self.max.y.max(vertex.y);
            self.max.z = self.max.z.max(vertex.z);
        }

        /// Merge two AABBs into one that encompasses both.
        pub fn merge(&self, other: &AABB) -> AABB {
            AABB {
                min: Vertex {
                    x: self.min.x.min(other.min.x),
                    y: self.min.y.min(other.min.y),
                    z: self.min.z.min(other.min.z),
                },
                max: Vertex {
                    x: self.max.x.max(other.max.x),
                    y: self.max.y.max(other.max.y),
                    z: self.max.z.max(other.max.z),
                },
            }
        }

        /// Check if a ray intersects this bounding box.
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

            let t_enter = t_min
                .0
                .min(t_max.0)
                .max(t_min.1.min(t_max.1))
                .max(t_min.2.min(t_max.2));
            let t_exit = t_max
                .0
                .max(t_min.0)
                .min(t_max.1.max(t_min.1))
                .min(t_max.2.max(t_min.2));

            t_enter <= t_exit && t_exit >= 0.0
        }
    }

    #[derive(Debug, Clone)]
    pub enum BVHNode {
        Leaf {
            bounding_box: AABB,
            triangle_indices: Vec<usize>, // Indices of triangles in the leaf
        },
        Internal {
            bounding_box: AABB,
            left: Arc<BVHNode>,
            right: Arc<BVHNode>,
        },
    }
    impl BVHNode {
        pub fn build(mesh: &Mesh, triangle_indices: Vec<usize>, depth: usize) -> BVHNode {
            // Base case: Create a leaf node if the triangle count is small.
            if triangle_indices.len() <= 2 {
                let mut bounding_box = AABB::new();
                for &index in &triangle_indices {
                    let triangle = &mesh.triangles[index];
                    for &vertex_index in &triangle.vertex_indices {
                        bounding_box.expand(&mesh.vertices[vertex_index]);
                    }
                }
                return BVHNode::Leaf {
                    bounding_box,
                    triangle_indices,
                };
            }

            // Compute the bounding box for the current set of triangles.
            let mut bounding_box = AABB::new();
            for &index in &triangle_indices {
                let triangle = &mesh.triangles[index];
                for &vertex_index in &triangle.vertex_indices {
                    bounding_box.expand(&mesh.vertices[vertex_index]);
                }
            }

            // Find the axis to split along (longest axis of the bounding box).
            let axis = {
                let extents = Vertex {
                    x: bounding_box.max.x - bounding_box.min.x,
                    y: bounding_box.max.y - bounding_box.min.y,
                    z: bounding_box.max.z - bounding_box.min.z,
                };
                if extents.x >= extents.y && extents.x >= extents.z {
                    0 // X-axis
                } else if extents.y >= extents.z {
                    1 // Y-axis
                } else {
                    2 // Z-axis
                }
            };

            // Sort triangles along the chosen axis.
            let mut sorted_indices = triangle_indices.clone();
            sorted_indices.sort_by(|&a, &b| {
                let center_a = mesh.triangles[a].center(&mesh.vertices);
                let center_b = mesh.triangles[b].center(&mesh.vertices);
                center_a[axis].partial_cmp(&center_b[axis]).unwrap()
            });

            // Split triangles into two groups.
            let mid = sorted_indices.len() / 2;
            let (left_indices, right_indices) = sorted_indices.split_at(mid);

            // Recursively build left and right subtrees.
            let left = Arc::new(BVHNode::build(mesh, left_indices.to_vec(), depth + 1));
            let right = Arc::new(BVHNode::build(mesh, right_indices.to_vec(), depth + 1));

            // Combine bounding boxes of left and right nodes.
            let bounding_box = left.bounding_box().merge(right.bounding_box());

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
    }

    impl BVHNode {
        pub fn intersect(&self, mesh: &Mesh, ray: &Ray) -> Option<(f64, usize)> {
            if !self.bounding_box().intersects(ray) {
                return None; // Ray misses this node.
            }

            match self {
                BVHNode::Leaf {
                    triangle_indices, ..
                } => {
                    let mut closest_hit: Option<(f64, usize)> = None;
                    for &index in triangle_indices {
                        let triangle = &mesh.triangles[index];
                        if let Some(t) = triangle.intersect(ray, &mesh.vertices) {
                            if closest_hit.is_none() || t < closest_hit.unwrap().0 {
                                closest_hit = Some((t, index)); // Store distance and triangle index.
                            }
                        }
                    }
                    closest_hit
                }
                BVHNode::Internal { left, right, .. } => {
                    let left_hit = left.intersect(mesh, ray);
                    let right_hit = right.intersect(mesh, ray);

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
    /////////////////////////////////////////////////////////////////////////
    // Ray are base on Vertex so camera for ray is also base on vertex it's
    // DIsplay related then Vertex are so used.
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
                .add(self.right.mul(pixel_camera_x))
                .add(self.up.mul(pixel_camera_y))
                .normalize(); // Normalize to get unit vector

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
