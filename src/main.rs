// load modules.
mod rust3d;
mod render_tools;
mod virtual_space;
mod models_3d;
// import dependencies.
use render_tools::rendering_object::{Mesh,Vertex};
use render_tools::visualization_v3::coloring::Color;
use rust3d::geometry::{CPlane,Point3d,Vector3d};
use virtual_space::{Virtual_space,DisplayPipeLine,Unit_scale,Display_config};
use std::sync::{Arc,Mutex};
//Start the main thread.
fn main(){
    // Init the first parameters.
    let display_parameters = Display_config::new( 3840 / 3,
        2160 / 3,
        Color::convert_rgb_color(157, 163, 170)
        );
    // Init the virtual space with multi thread concurrency context ( atomic reference counting and
    // mutual exclusion ).
    let vs = Arc::new(Mutex::new(Virtual_space::new("Project test", 
        Some("./geometry/torus.obj".to_owned()), 
        Unit_scale::Millimeters, 
        display_parameters)));
}

