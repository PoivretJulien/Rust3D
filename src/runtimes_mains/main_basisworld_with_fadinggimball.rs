use minifb::{Key, Window, WindowOptions};
mod render_tools;
mod models_3d;
mod rust3d;
mod virtual_space;
use core::f64;
use render_tools::rendering_object::{Mesh, Vertex};
use render_tools::visualization_v3::coloring::Color;
use render_tools::visualization_v3::Camera;
use iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::*;
use rust3d::transformation::rotate_z;
use rust3d::{draw::*, transformation};
use rust3d::{geometry::*, utillity};
use std::sync::{Arc,Mutex};
use std::time::Duration;
use virtual_space::*;
fn main() {
    // init the app with inputs parameters.
    const PATH: &str = "./geometry/ghost_b.obj";
    let mut program = Virtual_space::new(
        "first test",
        None,
        Unit_scale::Millimeters,
        Display_config::new(3840 /3 ,2160/3),
    );
    // import a mesh...
    let mut mesh = Mesh::new(); // pointer holder of the mesh structure
    if let Ok(obj) = Mesh::import_obj_with_normals(PATH) {
        println!("imported .obj file from: \"{PATH}\"");
        println!("obj-> Triangles number: {}", obj.triangles.len());
        println!("obj-> Vertices number: {}", obj.vertices.len());
        // pass inner pointers.
        mesh.vertices = obj.vertices;
        mesh.triangles = obj.triangles;
    } else {
        eprint!("Error on import.");
    }
    // init a CPlane.
    let origin = Point3d::new(1.156, -0.247, 1.245);
    let point_u = Point3d::new(1.012, 0.174, 1.081);
    let point_v = Point3d::new(0.854, -0.247, 1.510);
    // Create a construction plane from 3 points.
    let plane = CPlane::new_origin_x_aligned_y_oriented(&origin, &point_u, &point_v);
    // Transform with CPlane.
    mesh.vertices.par_iter_mut().for_each(|vert| {
        *vert = plane.point_on_plane(vert.x, vert.y, vert.z).to_vertex();
    });
    // Scale With matrix 4x3.
    let matrix = transformation::scaling_matrix_from_center_4x3(origin, 0.5, 0.5, 0.5);
    let v = transformation::transform_points_4x3(&matrix, &mesh.vertices);
    mesh.vertices = v; // change allocated pointers,the hold ones are free automatically.
    mesh.recompute_normals();
    ////////moved values////////////////////////////////////////////////////////
    // now the mesh and CPlane variables of above transfer the ownership to object of type Object3d.
    let object = 
    Arc::new(Mutex::new(Object3d::new(0, Some(Arc::new(Mutex::new(Displayable::Mesh(mesh)))), plane, 0.5)));
    println!("{}", object.lock().ok().unwrap());
    // test to add other type of Object3d //////////////////////////////////////
    program.add_obj(object); // transfer ownership to program.
    let vert = vec![Vertex::new(6.28, 1.6, 81.0)];
    let normal = Vector3d::new(0.0, 0.0, 1.0);
    let object2 = Arc::new(Mutex::new(Object3d::new(
        0,
        Some(Arc::new(Mutex::new(Displayable::Vertex(vert)))),
        CPlane::new(&origin, &normal),
        0.5,
    )));
    program.add_obj(object2);
    println!("{}", program);
    ////Draw a grid in world Center ////////////////////////////////////////////
    let pt_origin = Point3d::new(0.0, 0.0, 0.0);
    let x_dir = Vector3d::new(1.0, 0.0, 0.0);
    let z_dir = Vector3d::new(0.0, 0.0, 1.0);
    let mut world_plane = CPlane::new_normal_x_oriented(&pt_origin, &x_dir, &z_dir);
    let mut grid_pt = draw_3d_grid(&world_plane, 6.0, 6.0, 0.5);
    println!("System World Plane:{0}", world_plane);
    // Scale the Grid to display unit (future display pipeline responsibility)
    grid_pt.iter_mut().for_each(|pt| {
        *pt *= 0.1;
    });
    println!("Grid dimension: {0}x{0}", ((grid_pt.len()) as f64).sqrt());
    ////////////////////////////////////////////////////////////////////////////
    // Render the World Base System Coordinate./////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    const WIDTH: usize = 3840 / 3; // screen pixel width.
    const HEIGHT: usize = 2160 / 3; // screen pixel height.
    const BACK_GROUND_COLOR: u32 = 0x9da3aa; // 0x141314;
    const ANGLE_STEP: f64 = 0.3;
    let mut anim_transp = 0.0;
    let mut ramp = false;
    // Init a widows 2D mini buffer class.
    let mut window = Window::new(
        "Basis World System.",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        // panic on error (unwind stack and clean memory)
        panic!("{}", e);
    });
    ////////////////////////////////////////////////////////////////////////////
    // A simple allocated array of u32 initialized at 0
    // representing the color and the 2d position of points.
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    // Define the Display Unit Projection System.
    let camera = Camera::new(
        Point3d::new(0.0, -1.0, -0.27), // Camera position (1 is the max value)
        Point3d::new(0.0, 0.0, 0.0),    // Camera target ( relative to position must
        // be 0,0,0 )
        Vector3d::new(0.0, 0.0, 1.0), // Camera up vector (for inner cross product operation usually Y=1)
        WIDTH as f64,
        HEIGHT as f64,
        35.0,  // FOV (Zoom angle increase and you will get a smaller representation)
        0.5,   // Near clip plane
        100.0, // Far clip plane
    );
    let step = ANGLE_STEP.to_radians();
    window.set_target_fps(60); // limit to 60 fps.
    while window.is_open() && !window.is_key_down(Key::Escape) {
        for pixel in buffer.iter_mut() {
            *pixel = BACK_GROUND_COLOR; // Stet the bg color.
        }
        // Rotate the Grid.
        let mut index = 0usize;
        while index < grid_pt.len() {
            grid_pt[index] = rotate_z(grid_pt[index], step);
            index += 1;
        }
        let result_grid = camera.project_points(&grid_pt);
        // write Buffer.
        for data in result_grid.iter() {
            buffer[data.1 * WIDTH + data.0] =
                Color::convert_rgba_color(0, 0, 0, 1.0, BACK_GROUND_COLOR);
        }
        // animate the CPlane by rotation angle.
        world_plane.u = world_plane.u.rotate_around_axis(&world_plane.normal, -step);
        world_plane.v = world_plane.v.rotate_around_axis(&world_plane.normal, -step);
        if anim_transp< 3.14 && !ramp {
                anim_transp += step as f32;
            if anim_transp > 3.1 {
                ramp=!ramp;
            }
        }else {
            anim_transp -= step as f32;
            if anim_transp < 0.05 {
                ramp=!ramp;
            }
        } 
        let transparency_animation = utillity::ilerp(0.0, 3.14f32, anim_transp);
        draw_plane_gimball_3d(
            &mut buffer,
            WIDTH,
            world_plane,
            &camera,
            transparency_animation.unwrap(),
            BACK_GROUND_COLOR,
            0.04,
        );
        // Update buffer.
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
    }
    // mesh.export_to_obj_with_normals_fast("test.obj").ok();
}
