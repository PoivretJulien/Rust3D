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
    /*
     * Precision orientation test with matrix transformation scaling.
     * in a nice and elegant way.
     * */
    let mut mesh = Mesh::new(); // pointer holder of the mesh structure
    // Test of the new Mesh entity with CPlane.
    if let Ok(obj) = Mesh::import_obj_with_normals("./geometry/ghost_b.obj") {
       println!("obj-> Triangles number: {}",obj.triangles.len());
       println!("obj-> Vertices number: {}",obj.vertices.len());
       // pass inner pointers.
       mesh.vertices = obj.vertices;
       mesh.triangles = obj.triangles;
    }else{
        eprint!("Error on import.");
    }
    // the to_point3d should be optimized statically at compile time (zero cost abstraction).
    let origin = Vertex::new(1.156, -0.247,1.245).to_point3d();
    let point_u = Vertex::new(1.012, 0.174,1.081).to_point3d();
    let point_v = Vertex::new(0.854, -0.247,1.510).to_point3d();
    // Create a construction plane from 3 points.
    let plane = CPlane::new_origin_x_aligned_y_oriented(&origin, &point_u, &point_v);
    // Transform with CPlane.
    mesh.vertices.par_iter_mut().for_each(|vert|{
        *vert = plane.point_on_plane(vert.x, vert.y, vert.z).to_vertex();
    });
    // Scale With matrix 4x3
    let matrix = transformation::scaling_matrix_from_center_4x3(origin,0.5, 0.5, 0.5);
    let v = transformation::transform_points_4x3(&matrix, &mesh.vertices);
    mesh.vertices = v; // change allocated pointers,the hold ones are free automatically.
    mesh.recompute_normals();
    mesh.export_to_obj_with_normals_fast("test.obj").ok();
                                            
}
