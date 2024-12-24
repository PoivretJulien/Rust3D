use core::f64;

use minifb::{Key, Window, WindowOptions}; // render a 2d point in color on a defined screen size.

// My 3d lib for computational processing (resource: mcneel.com (for vector3d point3d), openia.com for basic 3d engine)
mod models_3d;
mod rust3d; // external file where 3d model(s) are conveniently stored.
use models_3d::NUKE_3D_MODEL;
use rayon::*;
use rust3d::draw::*;
use rust3d::geometry::{CPlane, Point3d, Vector3d}; // My rust Objects for computing 3d scalars.
use rust3d::transformation::*; // Basic 3d transformation of 3d Point.
use rust3d::utillity::*;
use rust3d::visualization::redering_object::Mesh;
use rust3d::visualization::*;
use std::cell::RefCell;
use std::rc::Rc; // a basic 3d engine plotting a 3d point on 2d screen.

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
    const DISPLAY_OBJ: bool = true;
    const DISPLAY_GRID: bool = true;
    const BACK_GROUND_COLOR: u32 = 0x141314;
    const DISPLAY_TEXT: bool = true;
    const DISPLAY_CIRCLE: bool = true;

    let mut import_obj = Vec::new();
    // .obj file importation test.
    if let Some(mesh) = read_obj_file("./geometry/dk.obj") {
        for vertex in mesh.vertices {
            import_obj.push(Point3d::new(vertex.x, vertex.y, vertex.z));
        }
        let ratio_dk = 0.2; //Dk scale ratio.
        let trans_vector = (0.5, -0.4, 0.0); //translation vector.
        pre_process_obj(
            &(trans_vector.0),
            &(trans_vector.1),
            &(trans_vector.2),
            &ratio_dk,
            &mut import_obj,
        );
    }

    // mutating a static memory involve unsafe code.
    // (i want to avoid that so i make a deep copy for the actual thread once not a big deal...)
    let mut is_that_really_a_nuke = NUKE_3D_MODEL.clone(); //Deep copy.

    if DISPLAY_NUKE {
        // Pre process 3d model before display.
        let trans_vector = (-0.75, 0.5, 0.0); //translation vector.
        pre_process_model(
            &(trans_vector.0),
            &(trans_vector.1),
            &(trans_vector.2),
            &DISPLAY_RATIO,
            &mut is_that_really_a_nuke,
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

    // A simple allocated array of u 32 initialized at 0
    // representing the color and the 2d position of points.
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    // Define the camera
    let camera = Camera::new(
        Point3d::new(0.0, 1.0, 0.25), // Camera position (1 is the max value)
        Point3d::new(0.0, 0.0, 0.0),  // Camera target (looking at the origin)
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

    // Define the world coordinates origin 3d point.
    let origin = Point3d::new(0.0, 0.0, 0.0);
    let mut grid = Vec::new();

    // Make a world grid.
    if DISPLAY_GRID {
        let grid_mv = Vector3d::new(5.0, -5.0, 0.0);
        let zaxis = Vector3d::new(0.0, 0.0, 1.0);
        let plane = CPlane::new(&origin, &zaxis); // Grid local coordinates.
        let xmax = 10.0; // Grid max x
        let ymax = 10.0; // Grid max y
        let unit = 5.0;
        grid = draw_3d_grid(&plane, &xmax, &ymax, &unit);
        for pt in grid.iter_mut() {
            (*pt) *= DISPLAY_RATIO;
            (*pt) += grid_mv * DISPLAY_RATIO;
        }
    }

    let mut circle = Vec::new();
    let pln = Vector3d::new(0.2, -0.2, 0.8);
    let plane_origin = Point3d::new(1.0 * 0.2, 0.0, 0.85 * DISPLAY_RATIO);
    let plane = CPlane::new(&plane_origin, &pln);
    if DISPLAY_CIRCLE {
        circle = draw_3d_circle(Point3d::new(0.0, 0.0, 0.0), 1.3, 400.0);
        for i in 0..circle.len(){
            unsafe { // Evaluate as safe in that context (no concurrent access). 
                let ptr = circle.as_ptr().offset(i as isize) as *mut Point3d;
                *ptr = plane
                    .point_on_plane_uv(&((*ptr).X * DISPLAY_RATIO), &((*ptr).Y * DISPLAY_RATIO));
            }
        }
    }

    // init memory for an animated 3d square.
    let mut moving_square = [origin; 4];
    let mut ct = 0usize; // memory 'cursor index'

    // mini frame buffer runtime class initialization.
    // loop infinitely until the windows is closed or the escape key is pressed.
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Clear the screen (0x0 = Black)
        for pixel in buffer.iter_mut() {
            *pixel = BACK_GROUND_COLOR; //set a dark gray color as background.
        }
        /////Draw Grid./////////////////////////////////////////////////////////
        if DISPLAY_GRID {
            for point in grid.iter() {
                let rotated_point = rotate_z(*point, angle);
                if let Some(projected_point) = camera.project(rotated_point) {
                    buffer[projected_point.1 * WIDTH + projected_point.0] =
                        Color::convert_rgba_color(255, 0, 255, 0.9, BACK_GROUND_COLOR);
                    //  mutate the buffer (we are in a single thread configuration).
                }
            }
        }
        /////Draw Text./////////////////////////////////////////////////////////

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
        ////Draw Circle (test).
        if DISPLAY_CIRCLE {
            for point in circle.iter() {
                let rotated_point = rotate_z(*point, angle);
                if let Some(projected_point) = camera.project(rotated_point) {
                    buffer[projected_point.1 * WIDTH + projected_point.0] =
                        Color::convert_rgba_color(0, 123, 244, 1.0, BACK_GROUND_COLOR);
                    //  mutate the buffer (we are in a single thread configuration).
                }
            }
        }
        /////////////////////////////////////////////////////////////
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
            display_nuke(
                &camera,
                &mut buffer,
                &WIDTH,
                &angle,
                &mut is_that_really_a_nuke,
            );
        }
        if DISPLAY_OBJ {
            display_obj(
                &camera,
                &mut buffer,
                &WIDTH,
                &angle,
                &mut import_obj,
                &BACK_GROUND_COLOR,
            );
        }
        // Compute Angle animation./////////////////////////////////////////////
        let step = 0.5; // step in degree.
        let degree = (angle * 360.0) / (f64::consts::PI * 2.0);
        if (degree) >= 359.0 {
            // prevent to panic in case of f64 overflow (subtraction will be optimized at compile time)
            angle = 0.0;
        } else {
            angle += degree_to_radians(&step); // increment angle rotation for the animation in loop
        }
        ////////////////////////////////////////////////////////////////////////
        // enjoy.
        if DISPLAY_TEXT {
            let (x, y) = ((WIDTH / 2) - 125, (HEIGHT / 2) + 100);
            let color = Color::convert_rgb_color(0, 250, 0);
            let text_height = 2usize;
            draw_text(
                &mut buffer,
                &HEIGHT,
                &WIDTH,
                &x,
                &y,
                format!("Rotating angle: {:05.1} Deg", degree).as_str(),
                &text_height,
                &color,
            );
        }
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap(); // update the buffer
    }
}

fn read_obj_file(path: &str) -> Option<Mesh> {
    use rust3d::visualization::redering_object::Mesh;
    let mesh_data = Mesh::count_obj_elements(path).unwrap();
    println!("\x1b[2J");
    println!("Importing .obj file (test) path:{0}", path);
    println!(
        "Mesh stat(s) before import Vertex(s):{0} Vertex Normal(s):{1} Triangle(s):{2})",
        mesh_data.0, mesh_data.1, mesh_data.2
    );
    if let Some(obj) = Mesh::import_obj_with_normals(path).ok() {
        println!(
            "After import: Triangles:{0}, Vertex(s):{1}",
            obj.triangles.len(),
            obj.vertices.len()
        );
        println!("Import success.");
        //obj.export_to_obj_with_normals_fast("./geometry/high_test.obj").ok();
        Some(obj)
    } else {
        None
    }
}

fn pre_process_model(
    trans_x: &f64,
    trans_y: &f64,
    trans_z: &f64,
    scale_ratio: &f64,
    model_3d: &mut [Point3d; 707],
) {
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
    mode_3d: &mut [Point3d; 707],
) {
    for p in mode_3d.iter_mut() {
        let pt_rotated = rotate_z(*p, *angle); // rotate selected 3d point.
                                               // use 3d engine to project point.
        if let Some(projected_point) = camera.project(pt_rotated) {
            // write withe pixel on 2d screen representing the rotating 3d model.
            buffer[projected_point.1 * width + projected_point.0] = 0xFFFFFF; // mutate the buffer (we are in a single thread configuration).
        }
    }
}

fn pre_process_obj(
    trans_x: &f64,
    trans_y: &f64,
    trans_z: &f64,
    scale_ratio: &f64,
    model_3d: &mut Vec<Point3d>,
) {
    for i in 0..model_3d.len() {
        model_3d[i].X += *trans_x;
        model_3d[i].Y += *trans_y;
        model_3d[i].Z += *trans_z;
        model_3d[i] *= *scale_ratio;
    }
}

use rust3d::visualization::coloring::Color;
fn display_obj(
    camera: &Camera,
    buffer: &mut Vec<u32>,
    width: &usize,
    angle: &f64,
    mode_3d: &mut Vec<Point3d>,
    back_ground_color: &u32,
) {
    for p in mode_3d.iter_mut() {
        let pt_rotated = rotate_z(*p, *angle); // rotate selected 3d point.
                                               // use 3d engine to project point.
        if let Some(projected_point) = camera.project(pt_rotated) {
            buffer[projected_point.1 * width + projected_point.0] =
                Color::convert_rgba_color(255, 255, 39, 0.93, *back_ground_color);
            //  mutate the buffer (we are in a single thread configuration).
        }
    }
}
