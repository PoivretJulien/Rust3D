#![feature(portable_simd)]
// Load modules.
use std::sync::{Arc,Mutex};
use rust3d::geometry::{Point3d,Vector3d,CPlane};
use rust3d::render_tools::rendering_object::Mesh;
use rust3d::virtual_space::{DisplayConfig,VirtualSpace,UnitScale,Object3d,Displayable,DisplayPipeLine};
use rust3d::render_tools::visualization_v3::coloring::Color;
// Import dependencies.
// Start the main thread.
fn main() {
    // Init the first parameters.
    let display_parameters =
        DisplayConfig::new( 1000, 1100, Color::convert_rgb_color(157, 163, 170));
    // Init the virtual space with multi thread concurrency context ( atomic reference counting and
    // mutual exclusion ).
    let vs = Arc::new(Mutex::new(VirtualSpace::new(
        "Project test",
        None, // no system file implemented for now.
        UnitScale::Millimeters,
        display_parameters,
    )));
    ////////////////////////////////////////////////////////////////////////////
    // Code some Geometry manipulation there.
    // Start by importing an object. (a torus test mesh in a .obj file.)
    let mut obj_mesh = Mesh::new();
    match Mesh::import_obj_with_normals("./geometry/ghost_b.obj") {
        Ok(obj) => {
            obj_mesh = obj;
        }
        Err(err) => {
            eprint!("{err}");
        }
    }
    let origin = Point3d::new(0.0, 0.0, 0.0);
    let normal = Vector3d::new(0.0, 0.0, 1.0);
    let x_direction = Vector3d::new(1.0, 0.0, 0.0);
    let construction_plane = CPlane::new_normal_x_oriented(&origin, &x_direction, &normal);
    // Create an Object3d can be (mesh,Vertex,Point3d,Vector3d).
    let object1 = Object3d::new(Displayable::Mesh(obj_mesh), construction_plane, 1.0);
    // vs.lock().unwrap().add_obj(object1); // first use of the Mutex so it can be accessed without additional check.
    if let Ok(mut mutex) = vs.lock() {
        // idiomatic way.
        mutex.add_obj(object1);
    }
    ////////////////////////////////////////////////////////////////////////////
    // Pass a copy of the visual space pointer.
    let mut display_conduit = DisplayPipeLine::new(vs.clone());
    display_conduit.start_display_pipeline();
}
