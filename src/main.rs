use minifb::{Key, Window, WindowOptions}; // render a 2d point in color on a defined screen size.

// My 3d lib for computational processing (resource: mcneel.com (for vector3d point3d), openia.com for basic 3d engine)
mod rust3d;
use rust3d::draw::*;
use rust3d::geometry::{Point3d, Vector3d}; // My rust Objects for computing 3d scalars.
use rust3d::transformation::*; // Basic 3d transformation of 3d Point.
use rust3d::visualization::*; // a basic 3d engine plotting a 3d point on 2d screen.

mod models_3d; // external file where 3d model(s) are conveniently stored.
use models_3d::NUKE_3D_MODEL;

// - a basic Rust program using CPU for animating 9 3d points on screen
//   representing a cube with a dot in the middle + 3 colors axis are
//   also represented a square of that cube with orange segments.
//   f32 would be enough and require for speed display. 
//   but... that is a lib for back end CAD applications
//   so a full double precision is used a f128 also is probably overkill... 

fn main() {
    /*
     * First projection of the rust 3d lib using Point3d and Vector3d Objects.
     */
    const WIDTH: usize = 800; // screen pixel width.
    const HEIGHT: usize = 600; // screen pixel height.
    const DISPLAY_RATIO: f64 = 0.109; // Display space model scale unit dimension.
    const DISPLAY_NUKE: bool = false; // Optional (for Graphical purpose).
    
    // mutating a static memory involve unsafe code.
    // (i want to avoid that so i make a deep copy for the actual thread once not a big deal...) 
    let mut is_that_really_a_nuke = NUKE_3D_MODEL.clone(); //Deep copy.

    if DISPLAY_NUKE {
        // Pre process 3d model before display.
        let trans_vector = (-0.75,0.5,0.0); //translation vector. 
        pre_process_model(
        &(trans_vector.0),
        &(trans_vector.1),
        &(trans_vector.2),
        &DISPLAY_RATIO,
        &mut is_that_really_a_nuke
        );
    }

    // Init a widows 2D mini buffer class.
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

    // a simple allocated array of u 32 initialized at 0
    // representing the color and the 2d position of points.
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    // Define the camera
    let camera = Camera::new(
        Vector3d::new(0.0, 1.0, 0.25), // Camera position (1 is the max value)
        Vector3d::new(0.0, 0.0, 0.0),  // Camera target (looking at the origin)
        Vector3d::new(0.0, 1.0, 0.0), // Camera up vector (for inner cross product operation usually Y=1)
        WIDTH as f64,
        HEIGHT as f64,
        35.0,  // FOV (Zoom angle increase and you will get a smaller representation)
        0.5,   // Near clip plane
        100.0, // Far clip plane
    );

    // Define a cube with 8 points (with a 9th point in the center)
    let mut points = vec![
        Point3d::new(0.0, 0.0, 0.0),
        Point3d::new(1.0, 0.0, 0.0),
        Point3d::new(0.0, 1.0, 0.0),
        Point3d::new(1.0, 1.0, 0.0),
        Point3d::new(0.0, 0.0, 1.0),
        Point3d::new(1.0, 0.0, 1.0),
        Point3d::new(0.0, 1.0, 1.0),
        Point3d::new(1.0, 1.0, 1.0),
        Point3d::new(0.50, 0.50, 0.50),
    ];

    // Translation Vector of the above 'cube'.
    let mv = Vector3d::new(-0.5, -0.5, 0.0);
    // Scale the 3d model to model space ratio and center it in the World Coordinates.
    for pt in points.iter_mut() {
        (*pt) *= DISPLAY_RATIO;
        (*pt) += mv * DISPLAY_RATIO;
    }

    let mut angle = 0.0; // Angle in radian.

    // init memory for an animated 3d square.
    let default_point = Point3d::new(0.0, 0.0, 0.0);
    let mut moving_square = [default_point; 4];
    let mut ct = 0usize; // memory 'cursor index'

    // mini frame buffer runtime class initialization.
    // loop infinitely until the windows is closed or the escape key is pressed.
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Clear the screen (0x0 = Black)
        for pixel in buffer.iter_mut() {
            *pixel = 0x0;
        }
        // Define the world coordinates origin 3d point.
        let origin = Point3d::new(0.0, 0.0, 0.0);

        // Project animated point on the 2d screen.
        // Compute only the animated 3d point in that loop.
        for (i, p) in points.iter().enumerate() {
            // walk the array of point (Vector<Point3d>)
            // and rotate a copy of the 3d Point by an  incremented angle value of 0.005 radians.
            let rotated_point = rotate_z(*p, angle);
            // Backup rotated square point for further drawing line after the computation of the animation (in the loop).
            if (i == 0) || (i == 1) || (i == 4) || (i == 5) {
                moving_square[ct] = rotated_point;
                ct += 1;
                if i == 5 {
                    ct = 0;
                }
            }
            // Un box projected point if a value is present an draw 3d points
            // and 2D Lines in projected space. (3d engine have already there completed it's task).
            // (a more fancy algorithm may use GPU for such projection operations rather than CPU based computation)
            if let Some(projected_point) = camera.project(rotated_point) {
                // Draw the point as a white pixel
                buffer[projected_point.1 * WIDTH + projected_point.0] = 0xFFFFFF;
            }
            // Draw world X and Y axis (they are rotating...)
            let rotated_x_axis = rotate_z(Point3d::new(1.0, 0.0, 0.0), angle) * DISPLAY_RATIO;
            draw_line(
                &mut buffer,
                WIDTH,
                camera.project(origin).unwrap(),
                camera.project(rotated_x_axis).unwrap(),
                0xFF0000, // red
            );
            let rotated_y_axis = rotate_z(Point3d::new(0.0, 1.0, 0.0), angle) * DISPLAY_RATIO;
            draw_line(
                &mut buffer,
                WIDTH,
                camera.project(origin).unwrap(),
                camera.project(rotated_y_axis).unwrap(),
                0x00FF00, // Green
            );
        }
        // Draw the static (not moving) z Axis in blue at the center.
        draw_line(
            &mut buffer,
            WIDTH,
            camera.project(origin).unwrap(),
            camera
                .project(Point3d::new(0.0, 0.0, 1.0 * DISPLAY_RATIO))
                .unwrap(),
            0x0000FF, // Blue.
        );
        // Draw a square (with a diagonal) in orange from the rotated 3d points in the previous loop.
        draw_line(
            &mut buffer,
            WIDTH,
            camera.project(moving_square[0]).unwrap(),
            camera.project(moving_square[1]).unwrap(),
            0xFF3C00, // Orange
        );
        draw_line(
            &mut buffer,
            WIDTH,
            camera.project(moving_square[0]).unwrap(),
            camera.project(moving_square[2]).unwrap(),
            0xFF3C00, // Orange
        );
        draw_line(
            &mut buffer,
            WIDTH,
            camera.project(moving_square[2]).unwrap(),
            camera.project(moving_square[3]).unwrap(),
            0xFF3C00, // Orange
        );
        draw_line(
            &mut buffer,
            WIDTH,
            camera.project(moving_square[3]).unwrap(),
            camera.project(moving_square[1]).unwrap(),
            0xFF3C00, // Orange
        );

        draw_line(
            &mut buffer,
            WIDTH,
            camera.project(moving_square[3]).unwrap(),
            camera.project(moving_square[0]).unwrap(),
            0xFF3C00, // Orange
        );
        // Display nuke if true;
        if DISPLAY_NUKE {
            display_nuke(&camera, &mut buffer, &WIDTH,  &angle,&mut is_that_really_a_nuke);
        }
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap(); // update the buffer
        if angle >= (std::f64::MAX - 0.005) {
            // prevent to panic in case of f64 overflow (subtraction will be optimized at compile time)
            angle = 0.0;
        } else {
            angle += 0.005; // increment angle rotation for the animation in loop
        } // enjoy.
    }
}

fn pre_process_model(trans_x:&f64,trans_y:&f64,trans_z:&f64,scale_ratio:&f64,model_3d:&mut [Point3d;707]) {
       for i in 0usize..707 {
        model_3d[i].X += *trans_x;
        model_3d[i].Y += *trans_y;
        model_3d[i].Z += *trans_z;
        model_3d[i] *= *scale_ratio;
    }
}

fn display_nuke(
    camera: &Camera,
    buffer: &mut Vec<u32>,
    width: &usize,
    angle: &f64,
    mode_3d:&mut [Point3d;707]
) {
    for p in mode_3d.iter_mut() {
        let pt_rotated = rotate_z(*p,*angle); // rotate selected 3d point.
        if let Some(projected_point) = camera.project(pt_rotated) {
            // use 3d engine to project point.
            buffer[projected_point.1 * width + projected_point.0] = 0xFFFFFF; // mutate the buffer (we are in a single thread configuration)
        }
    }
}
