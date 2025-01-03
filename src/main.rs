use minifb::{Key, Window, WindowOptions};
mod display_pipe_line;
mod models_3d;
mod rust3d;
mod virtual_space;
use display_pipe_line::rendering_object::{Mesh, Vertex};
use display_pipe_line::visualization_v3::coloring::Color;
use display_pipe_line::visualization_v3::Camera;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rust3d::{draw::*, transformation};
use rust3d::geometry::*;
use rust3d::transformation::rotate_z;
use std::time::Duration;
use virtual_space::*;
fn main() {
     // init the app with inputs parameters.
     let mut program  = Virtual_space::new("first test",
         None , 
         Unit_scale::Millimeters, 
         Display_config::new(800, 600)
         );
    // import a mesh...
    let mut mesh = Mesh::new(); // pointer holder of the mesh structure
    if let Ok(obj) = Mesh::import_obj_with_normals("./geometry/ghost_b.obj") {
       println!("obj-> Triangles number: {}",obj.triangles.len());
       println!("obj-> Vertices number: {}",obj.vertices.len());
       // pass inner pointers.
       mesh.vertices = obj.vertices;
       mesh.triangles = obj.triangles;
    }else{
        eprint!("Error on import.");
    }
    // init a Cplane.
    let origin = Point3d::new(1.156, -0.247,1.245);
    let point_u = Point3d::new(1.012, 0.174,1.081);
    let point_v = Point3d::new(0.854, -0.247,1.510);
    // Create a construction plane from 3 points.
    let plane = CPlane::new_origin_x_aligned_y_oriented(&origin, &point_u, &point_v);
    // Transform with CPlane.
    mesh.vertices.par_iter_mut().for_each(|vert|{
        *vert = plane.point_on_plane(vert.x, vert.y, vert.z).to_vertex();
    });
    // Scale With matrix 4x3.
    let matrix = transformation::scaling_matrix_from_center_4x3(origin,0.5, 0.5, 0.5);
    let v = transformation::transform_points_4x3(&matrix, &mesh.vertices);
    mesh.vertices = v; // change allocated pointers,the hold ones are free automatically.
    mesh.recompute_normals();
    ////////moved values////////////////////////////////////////////////////////
    // now the mesh and CPlane variables of above transfer the ownership to object of type Object3d.
    let object = Object3d::new(0, Some(Displayable::Mesh(mesh)),plane, 0.5);
    println!("{}",object); 
    ////////////////////////////////////////////////////////////////////////////
    program.add_obj(object);// transfer ownership to program.
    let vert = vec![Vertex::new(6.28,1.6,81.0)];
    let normal = Vector3d::new(0.0,0.0,1.0);
    let object2 = Object3d::new(0, Some(Displayable::Vertex(vert)),CPlane::new(&origin,&normal), 0.5);
    program.add_obj(object2);
    println!("{}",program); 

    // mesh.export_to_obj_with_normals_fast("test.obj").ok();
}
