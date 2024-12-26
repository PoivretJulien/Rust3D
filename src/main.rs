use minifb::{Key, Window, WindowOptions};
mod models_3d;
mod rust3d;
use rust3d::draw::*;
use rust3d::geometry::*;
use rust3d::transformation::rotate_z;
use rust3d::utillity::degree_to_radians;
use rust3d::visualization::coloring::Color;
use rust3d::visualization::redering_object::Mesh;
use rust3d::visualization_v2::Camera;

fn main() {
    const WIDTH: usize = 1470; // screen pixel width.
    const HEIGHT: usize = 956; // screen pixel height.
    const DISPLAY_RATIO: f64 = 0.57; // Display space model scale unit dimension.
    const BACK_GROUND_COLOR: u32 = 0x141314;
    const ANGLE_STEP: f64 = 3.0;
    const DISPLAY_CIRCLE: bool = true;
    let z_offset = Vector3d::new(0.0, 0.0, -0.485); //translation vector.
    println!("\x1b[2J");
    let mut import_obj = Vec::new();
    /////////IMPORT MESH (.obj file)///////////////////////////////////////////
    if let Some(mesh) = Mesh::import_obj_with_normals("./geometry/ghost_b.obj").ok() {
        println!(
            "\x1b[0;0HImported: Total:({0}) Vertex(s).",
            mesh.vertices.len()
        );
        /*
            for now switch to the Point3d format
            instead of Vertex (this is going to change soon).
        */
        for vertex in mesh.vertices {
            // Center and scale the Cloud of point.
            import_obj.push((Point3d::new(vertex.x, vertex.y, vertex.z) + z_offset) * DISPLAY_RATIO);
        }
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
    let mut circle = Vec::new();
    let pln = Vector3d::new(0.2, -0.2, 0.8);
    let plane_origin = Point3d::new(0.0, 0.0, 0.3 * DISPLAY_RATIO);
    let plane = CPlane::new(&plane_origin, &pln);
    if DISPLAY_CIRCLE {
        circle = draw_3d_circle(Point3d::new(0.0, 0.0, 0.0), 0.3, 800.0);
        for i in 0..circle.len() {
            unsafe {
                // Evaluate as safe in that context (no concurrent access).
                let ptr = circle.as_ptr().offset(i as isize) as *mut Point3d;
                *ptr = plane
                    .point_on_plane_uv(&((*ptr).X * DISPLAY_RATIO), &((*ptr).Y * DISPLAY_RATIO));
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////
    // A simple allocated array of u 32 initialized at 0
    // representing the color and the 2d position of points.
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    // Define the camera
    let camera = Camera::new(
        Point3d::new(0.0, 1.0, 0.3), // Camera position (1 is the max value)
        Point3d::new(0.0, 0.0, 0.0),  // Camera target (looking at the origin)
        Vector3d::new(0.0, 1.0, 0.0), // Camera up vector (for inner cross product operation usually Y=1)
        WIDTH as f64,
        HEIGHT as f64,
        35.0,   // FOV (Zoom angle increase and you will get a smaller representation)
        0.5,   // Near clip plane
        100.0, // Far clip plane
    );
    let step = degree_to_radians(&ANGLE_STEP);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Clear the screen (0x0 = Black)
        for pixel in buffer.iter_mut() {
            *pixel = BACK_GROUND_COLOR; //set a dark gray color as background.
        }
        ///// Compute Vertex Rotation ./////////////////////////////////////////
        let mut index = 0usize;
        while index < import_obj.len() {
            import_obj[index] = rotate_z(import_obj[index], step);
            index += 1;
        }
        if DISPLAY_CIRCLE {
            index = 0;
            while index < circle.len() {
                circle[index] = rotate_z(circle[index], step);
                index += 1;
            }
            let result = camera.project_points(&circle);
            for data in result.iter() {
                buffer[data.1 * WIDTH + data.0] =
                    Color::convert_rgba_color(0, 123, 244, 1.0, BACK_GROUND_COLOR);
                //  mutate the buffer (we are in a single thread configuration).
            }
        }
        // Compute projection.
        let result = camera.project_points(&import_obj);
        // Then write buffer.
        for data in result.iter() {
            buffer[data.1 * WIDTH + data.0] =
                Color::convert_rgba_color(255, 0, 255, 1.0, BACK_GROUND_COLOR);
        }
        let (x, y) = ((WIDTH / 2) - 490, (HEIGHT / 2) + 430);
        let color = Color::convert_rgb_color(0, 250, 0);
        let text_height = 2usize;
        draw_text(
            &mut buffer,
            &HEIGHT,
            &WIDTH,
            &x,
            &y,
            format!(
                "Call Of Duty Ghost Visualisation 2 :{0:?}",
                std::time::Instant::now().to_owned()
            )
            .as_str(),
            &text_height,
            &color,
        );

        // update buffer.
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap(); // update the buffer
    }
}
