//extern crate rust3d;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust3d::draw::{blend_colors, blend_colors_deprecated};
use rust3d::render_tools::rendering_object::{Mesh, Vertex};
fn benchmark_a(c: &mut Criterion) {
    c.bench_function("Double dot product va", |b| {
        b.iter(|| {
            let v_dir = Vertex::new(-2.0, -2.0, -2.0);
            let face_normal_a = Vertex::new(-2.0, -2.0, -2.0);
            let face_normal_b = Vertex::new(-2.0, -2.0, -2.0);
            Mesh::simd_dot_x2_va(
                &black_box(v_dir),
                &black_box(face_normal_a),
                &black_box(face_normal_b),
            )
        })
    });
    c.bench_function("Double dot product vb", |b| {
        b.iter(|| {
            let v_dir = black_box(Vertex::new(-2.0, -2.0, -2.0));
            let face_normal_a = Vertex::new(-2.0, -2.0, -2.0);
            let face_normal_b = Vertex::new(-2.0, -2.0, -2.0);
            Mesh::simd_dot_x2_vb(
                &black_box(v_dir),
                &black_box(face_normal_a),
                &black_box(face_normal_b),
            )
        })
    });
    c.bench_function("Double dot product classic (without simd instruction arrays)", |b| {
        b.iter(|| {
            let v_dir = black_box(Vertex::new(-2.0, -2.0, -2.0));
            let face_normal_a = Vertex::new(-2.0, -2.0, -2.0);
            let face_normal_b = Vertex::new(-2.0, -2.0, -2.0);
            let dot_x2 = |dir:&Vertex,normal_a:&Vertex,normal_b:&Vertex|->[f64;2]{
                [
                    (*dir)*(*normal_a),
                    (*dir)*(*normal_b),
                ]
            }; 
            dot_x2(&black_box(v_dir),&black_box(face_normal_a),&black_box(face_normal_b))
        })
    });
    c.bench_function(
        "Blend between two colors from an alpha value version 1 (classic readable coding))",
        |b| b.iter(|| blend_colors(black_box(0x141414), black_box(0x245625), black_box(0.5))),
    );
    c.bench_function(
        "Blend between two colors from an alpha value version 2 (inline assignation.)",
        |b| {
            b.iter(|| {
                blend_colors_deprecated(black_box(0x141414), black_box(0x245625), black_box(0.5))
            })
        },
    );
}
// Group benchmarks and run.
criterion_group!(benches, benchmark_a);
criterion_main!(benches);
