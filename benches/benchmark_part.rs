
//extern crate rust3d;
use rust3d::render_tools::rendering_object::{Vertex,Mesh};
use criterion::{criterion_main,criterion_group,Criterion,black_box};
fn benchmark_a(c:&mut Criterion){
    c.bench_function("Double dot product va", |b| { b.iter(||{ 
        let v_dir = Vertex::new(-2.0,-2.0,-2.0);
        let face_normal_a = Vertex::new(-2.0,-2.0,-2.0);
        let face_normal_b = Vertex::new(-2.0,-2.0,-2.0);
        Mesh::simd_dot_x2_va(&v_dir, &face_normal_a, &face_normal_b)})} );
    c.bench_function("Double dot product vb", |b| { b.iter(||{ 
        let v_dir = Vertex::new(-2.0,-2.0,-2.0);
        let face_normal_a = Vertex::new(-2.0,-2.0,-2.0);
        let face_normal_b = Vertex::new(-2.0,-2.0,-2.0);
        Mesh::simd_dot_x2_vb(&v_dir, &face_normal_a, &face_normal_b)})} );
}
// Group benchmarks and run.
criterion_group!(benches, benchmark_a);
criterion_main!(benches);

