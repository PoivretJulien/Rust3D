// load modules.
mod models_3d;
mod render_tools;
mod rust3d;
mod virtual_space;
// import dependencies.
use render_tools::rendering_object::{Mesh, Vertex};
use render_tools::visualization_v3::coloring::Color;
use rust3d::geometry::{CPlane, Point3d, Vector3d};
use std::sync::{Arc, Mutex};
use virtual_space::{
    DisplayPipeLine, Display_config, Displayable, Object3d, Unit_scale, Virtual_space,
};
//Start the main thread.
fn main() {
    // Init the first parameters.
    let display_parameters =
        Display_config::new( 600, 700, Color::convert_rgb_color(157, 163, 170));
    // Init the virtual space with multi thread concurrency context ( atomic reference counting and
    // mutual exclusion ).
    let vs = Arc::new(Mutex::new(Virtual_space::new(
        "Project test",
        None, // no system file implemented for now.
        Unit_scale::Millimeters,
        display_parameters,
    )));
    ////////////////////////////////////////////////////////////////////////////
    // Code some Geometry manipulation there.
    // Start by importing an object. (a torus test mesh in a .obj file.)
    let mut obj_mesh = Mesh::new();
    match Mesh::import_obj_with_normals("./geometry/dk.obj") {
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
