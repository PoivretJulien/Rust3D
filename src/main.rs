use minifb::{Key, Window, WindowOptions}; // render a 2d point in color on a defined screen size.

// My 3d lib for computational processing (resource: mcneel.com (for vector3d point3d), openia.com for basic 3d engine)
use rust_3d::draw::*;
use rust_3d::geometry::{Point3d, Vector3d}; // My rust Objects for computing 3d scalars.
use rust_3d::transformation::*; // Basic 3d treansformation of 3dPoint.
use rust_3d::visualization::*; // a basic 3d engine ploting a 3d point on 2d screen.

                               

// - a basic Rust program using CPU for animating 9 3d points on screen
//   representing a cube with a dot in the midle + 3 colors axies are 
//   also represented + one arbitrary segment in orange.
fn main() {
    /*
     * First projection of the rust_3d module 3d Point
     * with a very basic but mysterious 3d engine in rust.
     */
    const WIDTH: usize = 800;
    const HEIGHT: usize = 600;
    // Init a widows class.
    let mut window = Window::new(
        "Simple Camera 3D Projection",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        // panic on error (unwind stack and clean memory)
        panic!("{}", e);
    });
    // a simple allocated array of u32 initialized at 0
    // representing the color and the 2d position of points.
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    // Define the camera
    let camera = Camera::new(
        Vector3d::new(0.0, -1.0, 0.2), // Camera position
        Vector3d::new(0.0, 0.0, -0.1), // Camera target (looking at the origin)
        Vector3d::new(0.0, 0.0, 1.0),  // Camera up vector
        WIDTH as f64,
        HEIGHT as f64,
        35.0,  // FOV
        0.5,   // Near plane
        100.0, // Far plane
    );

    /*
     * for now render view dimension need to be calibrated (work with small values)
     */
    // Define a cube with 8 points (with a 9th point in the center)
    let points = vec![
        Point3d::new(0.0, 0.0, 0.0),
        Point3d::new(0.1, 0.0, 0.0),
        Point3d::new(0.0, 0.1, 0.0),
        Point3d::new(0.1, 0.1, 0.0),
        Point3d::new(0.0, 0.0, 0.1),
        Point3d::new(0.1, 0.0, 0.1),
        Point3d::new(0.0, 0.1, 0.1),
        Point3d::new(0.1, 0.1, 0.1),
        Point3d::new(0.05, 0.05, 0.05),
    ];

    let mut angle = 0.0; // Angle in radian.

    // mini frame buffer runtime class initialization.
    // loop infinitly until the windows is closed or the escape key is pressed.
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Clear the screen (0x0 = Black)
        for pixel in buffer.iter_mut() {
            *pixel = 0x0;
        }
        // Define 'static' origin 3d point
        let origin = Point3d::new(0.0, 0.0, 0.0);
        // Project animated point on the 2d screen.
        for (i, p) in points.iter().enumerate() {
            // walk the array of point (Vector<Point3d>)
            // and rotate a copy of the 3dPoint by an angle value.
            let rotated_point = rotate_z(*p, angle);
            // Unbox projected point if a value is present an draw 3d points
            // and Lines in 2d projected space. (3d engine have there completed it's task).
            // (a more fancy algorithm may use GPU for such operation rather than CPU)
            if let Some(projected_point) = camera.project(rotated_point) {
                // Draw the point as a white pixel
                buffer[projected_point.1 * WIDTH + projected_point.0] = 0xFFFFFF;
            }
            // Draw world X and Y axis (they are rotating...)
            if i == 1 {
                // if first point in Vector<Point3d> array.
                draw_line(
                    &mut buffer,
                    WIDTH,
                    camera.project(origin).unwrap(),
                    camera.project(rotated_point).unwrap(),
                    0x00FF00,
                );
            } else if i == 2 {
                // if third point in Vector<Point3d> array.
                draw_line(
                    &mut buffer,
                    WIDTH,
                    camera.project(origin).unwrap(),
                    camera.project(rotated_point).unwrap(),
                    0xFF0000,
                );
            } else if i == 8 {
                // if 9th (zero based) point in Vector<Point3d> array.
                // Draw an example line in orange.
                draw_line(
                    &mut buffer,
                    WIDTH,
                    camera.project(origin).unwrap(),
                    camera.project(rotated_point).unwrap(),
                    0xFA7600, // Orange color (see hexadecimal value).
                );
            }
        }
        // Draw the static (not moving) z Axis in blue.
        draw_line(
            &mut buffer,
            WIDTH,
            camera.project(origin).unwrap(),
            camera.project(Point3d::new(0.0, 0.0, 0.1)).unwrap(),
            0x0000FF,
        );
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap(); // update the buffer
        if angle >= (std::f64::MAX - 0.005) {
            // prevent to panic in case of f64 overflow (substraction will be optimized at compile time)
            angle = 0.0;
        } else {
            angle += 0.005; // increment angle rotation for the animation in loop
        } // enjoy.
    }
}

// ************************************************************************
// ******* First scratch of a basic computational component class *********
// ************************************************************************
#[allow(dead_code)]
mod rust_3d {
    pub mod geometry {
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
            ///  Create a 3d point.
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
                        / (f64::sqrt((*vector_a).X.powi(2) + (*vector_a).Y.powi(2) + (*vector_a).Z.powi(2))
                            * f64::sqrt(
                                (*vector_b).X.powi(2) + (*vector_b).Y.powi(2) + (*vector_b).Z.powi(2),
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
    pub mod intersection {
        use super::geometry::{Point3d, Vector3d};
        /// Compute intersection of two point by two vectors
        /// # Arguments 
        /// p1 first points (Point3d), d1 first direction (Vector3d)
        /// p2 first points (Point3d), d2 first direction (Vector3d)
        /// # Returns
        /// None if vectors never intersect or Point3d of intersection on Success.
        /// ***** this is a CAD version of the function using full fledged vectors feature. *****
        /// note: a perfomance drawing optimized function will be added just next.
        pub fn compute_intersection_cad(
            p1: &Point3d,
            d1: &Vector3d,
            p2: &Point3d,
            d2: &Vector3d,
        ) -> Option<Point3d> {
            let cross_d1_d2 = Vector3d::cross_product(d1, d2);
            let denom = cross_d1_d2 * cross_d1_d2; // dot product (squere of cross product vector)
            if f64::abs(denom) < 1e-10 {
                None // if lines never intersect.
            } else {
                let diff = *p2 - *p1; // Make vector delta.
                let t1 = Vector3d::cross_product(&diff, d2) * cross_d1_d2 / denom; // Compute intersection from formula. 
                Some(*p1 + ((*d1) * t1)) // Return result.
            }
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

            // Compute the view matrix
            // (transforms world coordinates to camera coordinates)
            // TODO: Vector3d struct use a menber 'Length' not required in this
            // Context.
            fn get_view_matrix_deprecated(&self) -> [[f64; 4]; 4] {
                let forward = Vector3d::new(
                    self.target.get_X() - self.position.get_X(),
                    self.target.get_Y() - self.position.get_Y(),
                    self.target.get_Z() - self.position.get_Z(),
                )
                .unitize_b();
                let right = Vector3d::cross_product(&forward, &self.up).unitize_b();
                let up = Vector3d::cross_product(&right, &forward).unitize_b();
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
            /// Same as above slightly faster.
            /// ( Vector Length is not automatically computed )
            /// - i have used manual dot product between Point3d and Vector3d
            /// without overhead cost at runtime..
            fn get_view_matrix(&self) -> [[f64; 4]; 4] {
                let forward = Vector3d::new(
                    self.target.get_X() - self.position.get_X(),
                    self.target.get_Y() - self.position.get_Y(),
                    self.target.get_Z() - self.position.get_Z(),
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
    }

    pub mod draw {
        // Bresenham's line algorithm.
        // Draw a line between two 2d points on screen.
        // it's a clever algorithm dynamicly plotting
        // the distance between two points.
        // - Bresenham's algorithm compute at each loop the direction (x,y)
        //   of next point to plot and so, draw a line in
        //   a 2d space and efficiently create the illusion of
        ///  a 3d line moving or rotating.
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

            // Compute absolute distance differance.
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
                // Write mutable buffer if inputs condition are met.
                if x >= 0 && x < width as isize && y >= 0 && y < (buffer.len() / width) as isize {
                    buffer[y as usize * width + x as usize] = color;
                }
                // Stop at end (when start point reach end point (x,y).
                if x == x1 && y == y1 {
                    break;
                }
                // Evaluate the next x and y direction (we are only in 2D space).
                // algorithm key point: is to  double the reminder by two
                // allowing to shrink the remider by dy and dx in one step
                // making Bresenham's line algorithm exceptionally efficient.
                // at chrinking the line towards the endpoint on x and y right direction
                // for a given orientation.
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
}

#[cfg(test)]
mod test {
    use super::rust_3d::geometry::*;
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
    use std::f64::consts::PI;
    #[test]
    fn test_vector3d_angle() {
        let v1 = Vector3d::new(0.0,1.0,0.0);
        let v2 = Vector3d::new(1.0,0.0,0.0);
        assert_eq!(
            PI / 2.0,
            Vector3d::compute_angle(&v1,&v2)
        );
    }
    use super::rust_3d::intersection::*;
    #[test]
    fn test_intersection_a(){
        let p1 = Point3d::new(0.0, 1.0, 0.0);
        let d1 = Vector3d::new(0.0,-1.0,0.0);
        let p2 = Point3d::new(1.0,0.0,0.0);
        let d2 = Vector3d::new(-1.0,0.0,0.0);
        assert_eq!(Point3d::new(0.0,0.0,0.0),compute_intersection_cad(&p1, &d1, &p2, &d2).unwrap());
    }
    #[test]
    fn test_intersection_b(){
        let p1 = Point3d::new(82.157, 30.323, 0.0);
        let d1 = Vector3d::new(0.643,-0.766,0.0);
        let p2 = Point3d::new(80.09,19.487,0.0);
        let d2 = Vector3d::new(0.94,0.342,0.0);
        let expected_result = Point3d::new(88.641361,22.59824,0.0); 
        if let Some(result) = compute_intersection_cad(&p1, &d1, &p2, &d2){
            if (result-expected_result).Length() < 26e-6
            {
                assert!(true);
            }else{
                assert!(false)
            }
        } 
    }
}
