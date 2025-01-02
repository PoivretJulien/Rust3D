use minifb::{Key, Window, WindowOptions};
mod display_pipe_line;
mod models_3d;
mod rust3d;
mod virtual_space;
use display_pipe_line::redering_object::{Mesh, Vertex};
use display_pipe_line::visualization_v3::coloring::Color;
use display_pipe_line::visualization_v3::Camera;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rust3d::draw::*;
use rust3d::geometry::*;
use rust3d::transformation::rotate_z;
use rust3d::utillity::degree_to_radians;
use std::time::Duration;

fn main() {
    /*
    Log 12/26/24:
         - V2 bring CPU parallelization & Camera movement via Matrix Transformation
           making a more convenient Api.
    - notes:
        Rust feel like the perfect low level language for operating machine with
        the safe spirit of ada designed for embeded system or real time operating
        machines (like medical x-ray), requiring a high level of reliability
        and security.
    - it's high level features language with low a level focus and optimization.
    */
    println!("\x1b[2J"); // Clear console.
                         // 3840x2160
                         // 1470x956
    const WIDTH: usize = 3840 / 3; // screen pixel width.
    const HEIGHT: usize = 2160 / 3; // screen pixel height.
    const DISPLAY_RATIO: f64 = 0.57; // Display space model scale unit dimension.
    const BACK_GROUND_COLOR: u32 = 0x141314;
    const ANGLE_STEP: f64 = 3.0;
    const DISPLAY_CIRCLE: bool = true;
    let mut pause: bool = false;
    let z_offset = Vertex::new(0.0, 0.0, -0.48); //-0.48 //translation vector.
    let mut import_obj = Vec::new();
    ///////////Triangle test////////////////////////////////////////////////////
    let mut tri = vec![
        (Vertex::new(0.05, 0.0, 0.0) + z_offset) * DISPLAY_RATIO,
        (Vertex::new(-0.05, 0.0, 0.0) + z_offset) * DISPLAY_RATIO,
        (Vertex::new(0.0, 0.0, 0.12) + z_offset) * DISPLAY_RATIO,
    ];
    ////////////////////////////////////////////////////////////////////////////

    /////////IMPORT MESH (.obj file)////////////////////////////////////////////
    if let Some(mut mesh) = Mesh::import_obj_with_normals("./geometry/ghost_b.obj").ok() {
        println!(
            "\x1b[0;0HImported: Total:({0}) Vertex(s) and ({1}) Triangle(s).",
            mesh.vertices.len(),
            mesh.triangles.len()
        );
        if mesh.is_watertight() {
            println!("Volume:({0})cubic/unit(s)", mesh.compute_volume());
        }
        mesh.vertices
            .par_iter_mut()
            .for_each(|v| *v = ((*v) + z_offset) * DISPLAY_RATIO);
        import_obj = mesh.vertices;
    }
    ////////////////////////////////////////////////////////////////////////////
    // Init a widows 2D mini buffer class.
    let mut window = Window::new(
        "Ghost Call of Duty Visualization Test.",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        // panic on error (unwind stack and clean memory)
        panic!("{}", e);
    });
    //// init a Circle representation on a CPlane.//////////////////////////////
    let mut preprocess_circle = Vec::new(); // since draw_3d_circle produce Point3d.
    let mut circle = Vec::new();
    let plane_normal = Vector3d::new(0.2, -0.2, 0.8);
    let plane_origin = Point3d::new(0.0, 0.0, 0.15 * DISPLAY_RATIO);
    let plane = CPlane::new(&plane_origin, &plane_normal);
    if DISPLAY_CIRCLE {
        preprocess_circle = draw_3d_circle(Point3d::new(0.0, 0.0, 0.0), 0.35, 400.0);
        for i in 0..preprocess_circle.len() {
            circle.push(
                plane
                    .point_on_plane_uv(
                        preprocess_circle[i].X * DISPLAY_RATIO,
                        preprocess_circle[i].Y * DISPLAY_RATIO,
                    )
                    .to_vertex(),
            );
            // unsafe {
            //     // Evaluate as safe in that context (no concurrent access).
            //     let ptr = circle.as_ptr().offset(i as isize) as *mut Point3d;
            //     *ptr = plane
            //         .point_on_plane_uv(&((*ptr).X * DISPLAY_RATIO), &((*ptr).Y * DISPLAY_RATIO));
            // }
        }
    }
    ////////////////////////////////////////////////////////////////////////////
    // A simple allocated array of u32 initialized at 0
    // representing the color and the 2d position of points.
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    // Define the Display Unit Projection System.
    let camera = Camera::new(
        Point3d::new(0.0, 1.0, 0.27), // Camera position (1 is the max value)
        Point3d::new(0.0, 0.0, 0.0),  // Camera target ( relative to position must
        // be 0,0,0 )
        Vector3d::new(0.0, 1.0, 0.0), // Camera up vector (for inner cross product operation usually Y=1)
        WIDTH as f64,
        HEIGHT as f64,
        35.0,  // FOV (Zoom angle increase and you will get a smaller representation)
        0.5,   // Near clip plane
        100.0, // Far clip plane
    );
    let step = degree_to_radians(&ANGLE_STEP);
    let target_frame_time = Duration::from_millis(1000 / 60); //limit 60 fps (for laptop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = std::time::Instant::now(); // record time  at t.
                                             // First format the screen background (0x0 = Black).
        for pixel in buffer.iter_mut() {
            *pixel = BACK_GROUND_COLOR; // set a dark gray color as background.
        }
        //Display circle////////////////////////////////////////////////////////
        let mut index = 0usize;
        if DISPLAY_CIRCLE {
            index = 0;
            while index < circle.len() {
                circle[index] = rotate_z(circle[index], step);
                index += 1;
            }
        }
        ///// Compute Vertex Rotation ./////////////////////////////////////////
        index = 0; // reset indexer.
        while index < import_obj.len() {
            import_obj[index] = rotate_z(import_obj[index], step);
            index += 1;
        }
        //  rotate triangle.
        tri[0] = rotate_z(tri[0], step);
        tri[1] = rotate_z(tri[1], step);
        tri[2] = rotate_z(tri[2], step);
        /////////////////////////////////////////////////////////////////////////
        /*
         *    notes:
         *    Pre process point via matrix to simulate camera movement
         *    for the unit projection system...
         *    (it's may not be the most efficient way but it's work.)
         *    tree versions though... one to mutate in place the other copy data
         *    for Undo and Redo operation stack.
         *    and the last one use combined matrix transformations of the original
         *    local space.
         */
        /////////////////////////////////////////////////////////////////////////
        const TEST_MOVE_SIMULATION: bool = false; // Switch on/off matrix test.
        let mut result = Vec::new();
        let mut result_circle = Vec::new();
        let mut result_tri = Vec::new();

        if TEST_MOVE_SIMULATION {
            // Camera Movement and Rotation Parameters.
            let forward_amount = 0.5; // Move forward (may be negative.)
            let yaw_angle = 0.0; // Rotate (in degrees around the Y-axis)
            let pitch_angle = 25.0; // Rotate (in degrees around the X-axis)

            ///////////////////////////////////////////////////////////
            // Generate the transformation matrix /////////////////////
            let transformation_matrix =
                camera.get_transformation_matrix_4x3(forward_amount, yaw_angle, pitch_angle);
            let transformation_matrix_pan = camera.get_pan_matrix_4x3(0.0, 0.135);
            let final_matrix = Camera::combine_matrices_4x3(vec![
                transformation_matrix,
                transformation_matrix_pan,
            ]);
            ///////////////////////////////////////////////////////////

            // Transform the points via transformation matrix.
            let t1 = camera.transform_points_4x3_par(&import_obj, final_matrix);
            let t2 = camera.transform_points_4x3_par(&circle, final_matrix);
            let t3 = camera.transform_points_4x3_par(&tri, final_matrix);

            // Raytrace //////////
            //-------------------
            // Then Compute Projection.
            // Compute projection case A.
            result = camera.project_points(&t1);
            result_circle = camera.project_points(&t2);
            result_tri = camera.project_points(&t3);
        } else {
            // Compute projection case B.
            result = camera.project_points(&import_obj);
            result_circle = camera.project_points(&circle);
            result_tri = camera.project_points(&tri);
        }
        // Then Rasterization of circle and geometry.
        for data in result_circle.iter() {
            buffer[data.1 * WIDTH + data.0] =
                Color::convert_rgba_color(0, 123, 244, 1.0, BACK_GROUND_COLOR);
        }
        for data in result.iter() {
            buffer[data.1 * WIDTH + data.0] =
                Color::convert_rgba_color(255, 0, 255, 1.0, BACK_GROUND_COLOR);
        }
        // Rasterize triangle (test).
        if result_tri.len() == 3 {
            use rust3d::draw::Point2D;
            let p0 = Point2D {
                x: result_tri[0].0 as f64,
                y: result_tri[0].1 as f64,
            };
            let p1 = Point2D {
                x: result_tri[1].0 as f64,
                y: result_tri[1].1 as f64,
            };
            let p2 = Point2D {
                x: result_tri[2].0 as f64,
                y: result_tri[2].1 as f64,
            };
            draw_triangle_optimized(
                &mut buffer,
                WIDTH,
                HEIGHT,
                &p0,
                &p1,
                &p2,
                Color::convert_rgba_color(255, 0, 0, 0.7, BACK_GROUND_COLOR),
            );
        }
        // Write infos feedback Text.
        let (x, y) = (1, 1);
        let color = Color::convert_rgb_color(0, 250, 0);
        let text_height = 1usize;
        draw_text(
            &mut buffer,
            &HEIGHT,
            &WIDTH,
            &x,
            &y,
            format!(
                "Visualisation system v3:{0:?}",
                std::time::Instant::now().to_owned()
            )
            .as_str(),
            &text_height,
            &color,
        );
        use minifb::KeyRepeat;
        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            pause = true;
        }
        if pause {
            //window.update();
            window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
            std::thread::sleep(Duration::from_secs(3)); // pause for 3 secs
            pause = false;
        } else {
            let elapsed = now.duration_since(now);
            // limit at 60 fps.
            // if computation is less than 16ms then sleep for: (1000ms/60) - elapsed.
            if elapsed < target_frame_time {
                std::thread::sleep(target_frame_time - elapsed);
            }
            // Update buffer.
            window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
        }
    }
}
