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
                .filter_map(|point| self.project(*point)) // Apply the project method in parallel
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
        pub fn project(&self, point: Vertex) -> Option<(usize, usize, f64)> {
            // Use precomputed matrices
            let camera_space_point = self.multiply_matrix_vector(self.view_matrix, point);

            // Extract depth in camera space (before projection)
            let depth_in_camera_space = camera_space_point.z;

            let projected_point =
                self.multiply_matrix_vector(self.projection_matrix, camera_space_point);

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
        pub fn project_maybe_outside(&self, point: Vertex) -> (f64, f64) {
            // Use precomputed matrices
            let camera_space_point = self.multiply_matrix_vector(self.view_matrix, point);

            let projected_point =
                self.multiply_matrix_vector(self.projection_matrix, camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.x / projected_point.z;
            let y = projected_point.y / projected_point.z;

            // Map the coordinates from [-1, 1] to screen space
            let screen_x = ((x + 1.0) * 0.5 * self.width) as isize;
            let screen_y = ((1.0 - y) * 0.5 * self.height) as isize;

            (screen_x as f64, screen_y as f64)
        }

        pub fn multiply_matrix_vector(&self, matrix: [[f64; 4]; 4], v: Vertex) -> Vertex {
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
                .map(|point| self.multiply_matrix_vector(transformation_matrix, *point))
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
                .map(|point| self.multiply_matrix_vector(transformation_matrix, *point))
                .collect()
        }

        /// Apply a transformation matrix to a mutable Vec<Vertex>
        pub fn transform_points_mut(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) {
            points.iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector(transformation_matrix, *point);
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
                *point = self.multiply_matrix_vector(transformation_matrix, *point);
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
                .map(|point| self.multiply_matrix_vector(pan_matrix, *point))
                .collect()
        }

        /// Apply panning to a set of mutable Vertex points.
        pub fn pan_points_mut(&self, points: &mut Vec<Vertex>, right_amount: f64, up_amount: f64) {
            // Step 1: Get the pan transformation matrix
            let pan_matrix = self.get_pan_matrix(right_amount, up_amount);

            // Step 2: Apply the pan transformation to all points
            points.iter_mut().for_each(|point| {
                (*point) = self.multiply_matrix_vector(pan_matrix, *point);
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
/*
 *   WORKINPROGRES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *   New mesh prototype with shared vertices.
 *   like this moving points location will also move
 *   faces.
 *   - triangles contain only index id of the vertex.
 *   - and normal vectors
 */
pub mod rendering_object {
    use crate::rust3d;
    use crate::rust3d::geometry::{CPlane, Point3d, Vector3d};
    use crate::rust3d::intersection::clip_line;
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
    use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};
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
                "Vertex(x: {:.2}, y: {:.2}, z: {:.2})",
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
    }
    ////////////////////////////////////////////////////////////////////////////
    // A temporary object representing a parametric object.
    // (this object will be merge in the implementation 
    //                                          of a classic mesh once designed )
    // maybe...
    ////////////////////////////////////////////////////////////////////////////
    pub struct MeshPlane {
        pub vertices: Vec<Vertex>,
        pub triangles: Vec<Triangle>,
    }

    impl MeshPlane {
        // Construct a plane mesh.
        pub fn new(
            buffer: &mut Vec<u32>,
            screen_width: usize,
            screen_height: usize,
            camera: &super::visualization_v3::Camera,
            matrix: Option<&[[f64; 4]; 3]>,
            construction_plane: &CPlane,
            u_length: f64,
            v_length: f64,
            spacing_unit: f64,
        ) {
            // Make a grid of points describing the plane and it's sides length
            // vertices count number.
            let uv_length = (
                ((u_length / spacing_unit) + std::f64::EPSILON) as usize,
                ((v_length / spacing_unit) + std::f64::EPSILON) as usize,
            );
            //println!("grid size:{:?}", uv_length,);
            let mut grid_points = vec![Vertex::new(0.0, 0.0, 0.0); uv_length.0 * uv_length.1];
            let mut pt_u = 0.0;
            let mut pt_v = 0.0;
            for v in 0..uv_length.1 {
                for u in 0..uv_length.0 {
                    grid_points[v * uv_length.0 + u] = (*construction_plane)
                        .point_on_plane_uv(pt_u, pt_v)
                        .to_vertex();
                    pt_u += spacing_unit;
                    if u == uv_length.0 - 1 {
                        pt_u = 0.0;
                    }
                }
                pt_v += spacing_unit;
            }
            // Apply transformations if needed.
            if let Some(m) = matrix {
                grid_points = 
                    rust3d::transformation::transform_points_4x3(m, &grid_points);
            }
            // Compute base vectors u and v for the actual plane.
            let plane_vector_u =
                grid_points[0 * uv_length.0 + 1] - grid_points[0 * uv_length.0 + 0];
            let plane_vector_v =
                grid_points[1 * uv_length.0 + 0] - grid_points[0 * uv_length.0 + 0];
            // Build Triangles.
            for u in 0..uv_length.0 {
                for v in 0..uv_length.1{
                    // quad origin point.
                    let vert_a = grid_points[v * uv_length.0 + u];
                    // quad point on u direction.
                    let vert_b = vert_a + plane_vector_u;
                    // quad point diagonal u+v direction.
                    let vert_c =
                        grid_points[v * uv_length.0 + u] + (plane_vector_u + plane_vector_v);
                    // quad point on v direction.
                    let vert_d = grid_points[v * uv_length.0 + u] + (plane_vector_v);
                    // temporary display of the indexing logic.
                    let p1 = camera.project_maybe_outside(vert_a);
                    let p2 = camera.project_maybe_outside(vert_b);
                    let p3 = camera.project_maybe_outside(vert_c);
                    let p4 = camera.project_maybe_outside(vert_d);
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
                }
            }
            // let tr = Triangle::with_indices(v0, v1, v2, vertices);
            //unimplemented!();
        }
    }
    ////////////////////////////////////////////////////////////////////////////

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

use crate::virtual_space::*;
use std::sync::Arc;
