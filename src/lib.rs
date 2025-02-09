#![feature(portable_simd)]
pub mod fonts_txt;
pub mod render_tools;
pub mod cad_operations;
pub mod virtual_space;

#[cfg(test)]
mod test {
    use super::cad_operations::geometry::*;
    use super::cad_operations::intersection::*;
    use super::cad_operations::transformation::project_3d_point_on_plane;
    use crate::render_tools::rendering_object::*;
    use crate::render_tools::rendering_object::{Mesh, Triangle, Vertex};
    use crate::render_tools::visualization_v3::coloring::Color;
    use core::f64;
    use std::f64::consts::PI;
    #[test]
    fn test_cross_product() {
        let vec_a = Vector3d::new(1.0, 0.0, 0.0);
        let vec_b = Vector3d::new(0.0, 1.0, 0.0);
        assert_eq!(
            Vector3d::new(0.0, 0.0, 1.0),
            Vector3d::cross_product(&vec_a, &vec_b)
        );
    }
    #[test]
    fn test_vector3d_negation() {
        let v_totest = Vector3d::new(0.0, -0.35, 8.0);
        let v_tocompare = Vector3d::new(0.0, 0.35, -8.0);
        assert_eq!(v_totest, -v_tocompare);
    }
    #[test]
    fn test_vector3d_length() {
        assert_eq!(f64::sqrt(2.0), Vector3d::new(1.0, 1.0, 0.0).Length());
    }
    #[test]
    fn test_vector3d_unitize() {
        let mut vector = Vector3d::new(6.0, 2.0, 8.0);
        vector.unitize();
        assert_eq!(1.0, vector.Length());
    }
    #[test]
    fn test_vector3d_scalar_a() {
        let mut vector = Vector3d::new(6.0, 2.0, 8.0);
        vector.unitize();
        vector = vector * 4.0;
        assert_eq!(4.0, vector.Length());
    }
    #[test]
    fn test_vector3d_scalar_b() {
        let mut vector = Vector3d::new(6.0, 2.0, 8.0);
        vector.unitize();
        vector *= 4.0;
        assert_eq!(4.0, vector.Length());
    }

    #[test]
    fn test_vector_is_same_direction() {
        let v1 = Vector3d::new(0.426427, -0.904522, 0.0);
        let v2 = Vector3d::new(0.688525, -0.7255212, 0.0);
        assert!(v1.is_same_direction(&v2, 0.94));
    }
    #[test]
    fn test_vector3d_angle() {
        let v1 = Vector3d::new(0.0, 1.0, 0.0);
        let v2 = Vector3d::new(1.0, 0.0, 0.0);
        assert_eq!(PI / 2.0, Vector3d::compute_angle(&v1, &v2));
    }
    #[test]
    fn test_intersection_a() {
        let p1 = Point3d::new(0.0, 1.0, 0.0);
        let d1 = Vector3d::new(0.0, -1.0, 0.0);
        let p2 = Point3d::new(1.0, 0.0, 0.0);
        let d2 = Vector3d::new(-1.0, 0.0, 0.0);
        assert_eq!(
            Point3d::new(0.0, 0.0, 0.0),
            compute_intersection_coplanar(&p1, &d1, &p2, &d2).unwrap()
        );
    }
    #[test]
    fn test_intersection_b() {
        let p1 = Point3d::new(82.157, 30.323, 0.0);
        let d1 = Vector3d::new(0.643, -0.766, 0.0);
        let p2 = Point3d::new(80.09, 19.487, 0.0);
        let d2 = Vector3d::new(0.94, 0.342, 0.0);
        let expected_result = Point3d::new(88.641361, 22.59824, 0.0);
        if let Some(result) = compute_intersection_coplanar(&p1, &d1, &p2, &d2) {
            if (result - expected_result).Length() < 26e-6 {
                assert!(true);
            } else {
                assert!(false)
            }
        }
    }
    #[test]
    fn test_coplanar_vectors_a() {
        let pt1 = Point3d::new(82.832047, 36.102125, -3.214695);
        let pt2 = Point3d::new(85.341596, 34.653236, -2.539067);
        let pt3 = Point3d::new(82.0, 34.0, -4.040822);
        let pt4 = Point3d::new(85.0, 34.0, -2.82932);
        let v1 = pt2 - pt1;
        let v2 = pt4 - pt3;
        let v3 = pt3 - pt1;
        assert_eq!(true, Vector3d::are_coplanar_b(&v1, &v2, &v3));
    }

    #[test]
    fn test_coplanar_vectors_b() {
        let pt1 = Point3d::new(82.832047, 36.102125, -3.214695);
        let pt2 = Point3d::new(85.341596, 34.653236, -2.139067); // (this point is changed)
        let pt3 = Point3d::new(82.0, 34.0, -4.040822);
        let pt4 = Point3d::new(85.0, 34.0, -2.82932);
        let v1 = pt2 - pt1;
        let v2 = pt4 - pt3;
        let v3 = pt3 - pt1;
        assert_eq!(false, Vector3d::are_coplanar_b(&v1, &v2, &v3));
    }

    #[test]
    fn test_rotated_vector_a() {
        let vector_a = Vector3d::new(0.0, 1.0, 0.0);
        let axis = Vector3d::new(0.0, 0.0, 1.0);
        let rotated_vector = vector_a.rotate_around_axis(&axis, PI / 4.0);
        // make a unit 45 deg vector.
        let expected_vector = Vector3d::new(1.0, 1.0, 0.0).unitize_b();
        if (expected_vector - rotated_vector).Length().abs() < 1e-6 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_rotated_vector_b() {
        let vector_a = Vector3d::new(-5.0, 4.0, 3.0);
        let axis = Vector3d::new(30.0, 3.0, 46.0);
        let rotated_vector = vector_a.rotate_around_axis(&axis, 1.047198);
        // make a unit 45 deg vector.
        let expected_vector = Vector3d::new(0.255535, 7.038693, -0.625699);
        //assert_eq!(rotated_vector,expected_vector);
        if (expected_vector - rotated_vector).Length().abs() < 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_vector3d_are_perpendicular() {
        let vec_a = Vector3d::new(1.3, 1.55, 2.4);
        let vec_b = Vector3d::new(0.9, 1.25, 1.11);
        let vec_c = Vector3d::cross_product(&vec_a, &vec_b).unitize_b();
        assert!(Vector3d::are_perpandicular(&vec_a, &vec_c));
    }

    #[test]
    fn test_project_point_on_plane() {
        let plane = [
            Point3d::new(0.0, 4.0, 7.0),
            Point3d::new(6.775301, 11.256076, 5.798063),
            Point3d::new(-2.169672, 21.088881, 9.471482),
            Point3d::new(-5.0, 9.0, 9.0),
        ];
        let pt_to_project = Point3d::new(-4.781863, 14.083874, 1.193872);
        let expected_result = Point3d::new(-2.571911, 13.271809, 8.748913);
        //assert_eq!(expected_result,project_3d_point_on_infinite_plane(&pt_to_project, &plane).unwrap());
        if let Some(point) = project_3d_point_on_plane(&pt_to_project, &plane) {
            if (point - expected_result).Length() < 1e-6 {
                assert!(true);
            } else {
                assert!(false);
            }
        }
    }
    #[test]
    fn project_vector_on_cplane() {
        let origin = Point3d::new(9.35028, 4.160783, -12.513656);
        let normal = Vector3d::new(-0.607828, -0.292475, -0.738244);
        let plane = CPlane::new(&origin, &normal);
        let vector = Vector3d::new(-1.883283, 2.49779, -6.130442);
        let expected_result = Point3d::new(10.469624, 8.103378, -14.997222);
        ////////////////////////////////////////////////////////////////////////
        // assert_eq!(expected_result,origin + vector.project_on_cplane(&plane));
        let result = origin + vector.project_on_cplane(&plane);
        if (result - expected_result).Length() <= 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
        ////////////////////////////////////////////////////////////////////////
    }

    #[test]
    fn test_if_point_is_on_plane() {
        let plane = [
            Point3d::new(0.0, 4.0, 7.0),
            Point3d::new(6.775301, 11.256076, 5.798063),
            Point3d::new(-2.169672, 21.088881, 9.471482),
            Point3d::new(-5.0, 9.0, 9.0),
        ];
        let point_to_test = Point3d::new(-2.571911, 13.271809, 8.748913);
        assert_eq!(true, point_to_test.is_on_plane(&plane));
    }

    #[test]
    fn test_local_point() {
        let plane_origin_pt = Point3d::new(4.330127, 10.0, -15.5);
        let plane_normal = Vector3d::new(0.5, 0.0, -0.866025);
        let point = Point3d::new(5.0, 5.0, 5.0);
        let expected_result = Point3d::new(2.5, 5.0, -22.330127);
        let cp = CPlane::new(&plane_origin_pt, &plane_normal);
        let result = cp.point_on_plane(point.X, point.Y, point.Z);
        // assert_eq!(expected_result,result);
        if (expected_result - result).Length().abs() <= 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_degrees_to_radians() {
        use super::cad_operations::utillity::*;
        let angle_to_test = 90.0;
        assert_eq!(f64::consts::PI / 2.0, degree_to_radians(&angle_to_test));
    }

    #[test]
    fn test_radians_to_degrees() {
        use super::cad_operations::utillity::*;
        let angle_to_test = f64::consts::PI / 2.0;
        assert_eq!(90.0, radians_to_degree(&angle_to_test));
    }

    #[test]
    fn test_ray_trace_v_a() {
        use super::cad_operations::intersection::*;
        let point = Point3d::new(7.812578, 4.543698, 23.058283);
        let direction = Vector3d::new(-1.398849, 0.106953, -0.982613);
        let plane_origin = Point3d::new(-8.015905, -1.866453, 5.80651);
        let plane_normal = Vector3d::new(0.65694, -0.31293, 0.685934);
        let plane = CPlane::new(&plane_origin, &plane_normal);
        let expected_result = Point3d::new(-9.583205, 5.873738, 10.838721);
        // assert_eq!(expected_result,intersect_line_with_plane(&point, &direction, &plane).unwrap());
        if let Some(result_point) = intersect_ray_with_plane(&point, &direction, &plane) {
            if (result_point - expected_result).Length().abs() < 1e-5 {
                assert!(true);
            } else {
                assert!(false);
                assert_eq!(result_point, expected_result);
            }
        } else {
            assert!(false);
        }
    }
    #[test]
    fn test_ray_trace_v_b() {
        use super::cad_operations::intersection::*;
        let point = Point3d::new(15.417647, 4.098069, 11.565836);
        let direction = Vector3d::new(-6.509447, -2.89155, -3.065556);
        let plane_origin = Point3d::new(-5.598372, -15.314516, -6.014116);
        let plane_normal = Vector3d::new(0.887628, 0.298853, 0.350434); //plane_normal = -plane_normal;
        let plane = CPlane::new(&plane_origin, &plane_normal);
        let expected_result = Point3d::new(-10.410072, -7.374817, -0.597459);
        // assert_eq!(expected_result,intersect_line_with_plane(&point, &direction, &plane).unwrap());
        if let Some(result_point) = intersect_ray_with_plane(&point, &direction, &plane) {
            if (result_point - expected_result).Length().abs() < 1e-4 {
                assert!(true);
            } else {
                assert_eq!(result_point, expected_result);
            }
        } else {
            assert!(false);
        }
    }
    #[test]
    fn test_color() {
        let red: u8 = 20;
        let green: u8 = 19;
        let blue: u8 = 20;
        assert_eq!(0x141314, Color::convert_rgb_color(red, green, blue));
    }
    #[test]
    fn test_import_export_obj_size_ligh() {
        let vertices = vec![
            Vertex::new(0.0, 0.0, 0.0),
            Vertex::new(1.0, 0.0, 1.0),
            Vertex::new(1.0, 1.0, 0.0),
            Vertex::new(0.0, 1.0, 0.2),
        ];
        let triangles = vec![
            Triangle::new(&vertices, [0, 1, 2]),
            Triangle::new(&vertices, [0, 2, 3]),
        ];
        let mesh = Mesh::new_with_data(vertices, triangles);
        mesh.export_to_obj_with_normals_fast("./geometry/exported_light_with_rust.obj")
            .ok();
        let expected_data = Mesh::count_obj_elements("./geometry/exported_light_with_rust.obj")
            .ok()
            .unwrap();
        let imported_mesh =
            Mesh::import_obj_with_normals("./geometry/exported_light_with_rust.obj").unwrap();
        assert_eq!(
            (expected_data.0, expected_data.2),
            (imported_mesh.vertices.len(), imported_mesh.triangles.len())
        );
    }
    #[test]
    fn test_import_export_obj_size_medium() {
        let expected_data = Mesh::count_obj_elements("./geometry/medium_geometry.obj")
            .ok()
            .unwrap();
        let imported_mesh =
            Mesh::import_obj_with_normals("./geometry/medium_geometry.obj").unwrap();
        imported_mesh
            .export_to_obj_with_normals_fast("./geometry/medium_geometry_exported_from_rust.obj")
            .ok();
        assert_eq!(
            (expected_data.0, expected_data.2),
            (imported_mesh.vertices.len(), imported_mesh.triangles.len())
        );
    }
    #[test]
    fn test_import_obj_size_hight() {
        let expected_data = Mesh::count_obj_elements("./geometry/hight_geometry.obj")
            .ok()
            .unwrap();
        let imported_mesh = Mesh::import_obj_with_normals("./geometry/hight_geometry.obj").unwrap();
        imported_mesh
            .export_to_obj_with_normals_fast("./geometry/hight_geometry_exported_from_rust.obj")
            .ok();
        assert_eq!(
            (expected_data.0, expected_data.2),
            (imported_mesh.vertices.len(), imported_mesh.triangles.len())
        );
    }
    #[test]
    fn test_nurbs_curve_a() {
        let cv = vec![
            Point3d::new(0.0, 0.0, 0.0),
            Point3d::new(0.0, 10.0, 0.0),
            Point3d::new(10.0, 10.0, 0.0),
            Point3d::new(10.0, 0.0, 0.0),
            Point3d::new(20.0, 0.0, 0.0),
            Point3d::new(20.0, 10.0, 0.0),
        ];
        let crv = NurbsCurve::new(cv, 5);
        assert_eq!(Point3d::new(10.0, 5.0, 0.0), crv.evaluate(0.5));
        let pt = crv.evaluate(0.638);
        let expected_result = Point3d::new(13.445996, 3.535805, 0.0);
        if (pt - expected_result).Length() < 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    #[test]
    fn test_nurbs_curve_b() {
        let cv = vec![
            Point3d::new(-42.000, -13.000, 0.000),
            Point3d::new(-27.300, -26.172, 22.000),
            Point3d::new(-23.034, 15.562, 0.000),
            Point3d::new(7.561, -14.082, -21.000),
            Point3d::new(8.000, -19.000, 0.000),
        ];
        let crv = NurbsCurve::new(cv, 4);
        let t = 0.3;
        if (21.995 - crv.radius_of_curvature(t)).abs() < 1e-3 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_triangle_area() {
        // The following Triangle is flat in XY plane.
        let v1 = Vertex::new(1.834429, 0.0, -0.001996);
        let v2 = Vertex::new(1.975597, 0.0, 0.893012);
        let v3 = Vertex::new(2.579798, 0.0, 0.150466);
        let vertices = vec![v1, v2, v3];
        let tri = Triangle::new(&vertices, [0, 1, 2]);
        let expected_reuslt_area = 0.322794;
        let result = tri.get_triangle_area(&vertices);
        if (expected_reuslt_area - result).abs() < 1e-6 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn volume_mesh_test() {
        let obj = Mesh::import_obj_with_normals("./geometry/mesh_volume.obj").unwrap();
        let volume = obj.compute_volume();
        let expected_volume = 214.585113;
        if (volume - expected_volume).abs() < 1e-5 {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn volume_mesh_water_tight_a() {
        let obj = Mesh::import_obj_with_normals("./geometry/mesh_volume.obj").unwrap();
        if obj.is_watertight() {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    #[test]
    fn volume_mesh_water_tight_b() {
        let obj = Mesh::import_obj_with_normals("./geometry/non_water_tight.obj").unwrap();
        if !obj.is_watertight() {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    #[test]
    fn volume_mesh_water_tight_c_parrallel() {
        let obj = Mesh::import_obj_with_normals("./geometry/non_water_tight.obj").unwrap();
        if !obj.is_watertight_par() {
            assert!(true);
        } else {
            assert!(false);
        }
        let obj2 = Mesh::import_obj_with_normals("./geometry/mesh_volume.obj").unwrap();
        if obj2.is_watertight_par() {
            assert!(true);
        } else {
            assert!(false);
        }
    }
    #[test]
    fn test_inside_a_mesh() {
        let obj = Mesh::import_obj_with_normals("./geometry/torus.obj").unwrap();
        let v1_to_test = Vertex::new(0.0, 0.0, 0.0); // should be outside the torus.
        let v2_to_test = Vertex::new(4.0, 0.0, 0.0); // should be inside.
        assert_eq!(false, v1_to_test.is_inside_a_mesh(&obj));
        assert_eq!(true, v2_to_test.is_inside_a_mesh(&obj));
    }
    #[test]
    fn test_points_are_colinear() {
        // point 8 to 17 are a line (zero based).
        let p_tarray = vec![
            Point3d::new(1.575, 2.077, 1.777),
            Point3d::new(1.672, 2.240, 1.732),
            Point3d::new(1.765, 2.398, 1.663),
            Point3d::new(1.854, 2.549, 1.576),
            Point3d::new(1.940, 2.692, 1.474),
            Point3d::new(2.023, 2.828, 1.360),
            Point3d::new(2.104, 2.956, 1.236),
            Point3d::new(2.184, 3.077, 1.105),
            Point3d::new(2.276, 3.166, 0.961),
            Point3d::new(2.367, 3.255, 0.818),
            Point3d::new(2.459, 3.344, 0.675),
            Point3d::new(2.551, 3.433, 0.531),
            Point3d::new(2.643, 3.522, 0.388),
            Point3d::new(2.734, 3.611, 0.245),
            Point3d::new(2.826, 3.700, 0.102),
            Point3d::new(2.918, 3.789, -0.042),
            Point3d::new(3.010, 3.878, -0.185),
            Point3d::new(3.101, 3.967, -0.328),
            Point3d::new(3.193, 4.056, -0.472),
            Point3d::new(3.321, 4.120, -0.608),
            Point3d::new(3.459, 4.179, -0.738),
            Point3d::new(3.607, 4.232, -0.857),
            Point3d::new(3.767, 4.280, -0.964),
            Point3d::new(3.938, 4.321, -1.056),
            Point3d::new(4.118, 4.355, -1.128),
            Point3d::new(4.307, 4.383, -1.180),
            Point3d::new(4.502, 4.405, -1.208),
            Point3d::new(4.699, 4.422, -1.213),
            Point3d::new(4.896, 4.435, -1.194),
            Point3d::new(5.089, 4.446, -1.154),
            Point3d::new(5.277, 4.457, -1.094),
            Point3d::new(5.459, 4.468, -1.015),
            Point3d::new(5.632, 4.483, -0.921),
            Point3d::new(5.796, 4.501, -0.812),
            Point3d::new(5.951, 4.524, -0.691),
            Point3d::new(6.095, 4.553, -0.559),
            Point3d::new(6.228, 4.589, -0.417),
            Point3d::new(6.350, 4.633, -0.266),
            Point3d::new(6.458, 4.684, -0.109),
            Point3d::new(6.554, 4.743, 0.054),
        ];
        if Point3d::are_points_collinear(&p_tarray[8..19], 1e-3) {
            assert!(true);
        } else {
            assert!(false);
        }
        if !Point3d::are_points_collinear(&p_tarray[16..35], 1e-3) {
            assert!(true);
        } else {
            assert!(false);
        }
        // find the 3 first segments where the points array describe a straight line.
        if let Some(result) = Point3d::find_first_collinear_points(&p_tarray[0..], 1e-3) {
            assert_eq!((7, 10), result);
        }
    }
    #[test]
    fn test_ray_trace() {
        let obj = Mesh::import_obj_with_normals("./geometry/flatbox.obj")
            .ok()
            .unwrap();
        let v_inside = Vertex::new(-0.130, -0.188, 2.327);
        assert!(v_inside.is_inside_a_mesh(&obj));
        let v_outside = Vertex::new(0.623, -0.587, 2.327);
        assert!(!v_outside.is_inside_a_mesh(&obj));

        let pt_origin = Vertex::new(1.240, -0.860, 3.169);
        let pt_direction = Vertex::new(-0.743, 0.414, -0.526);

        let ray = Ray::new(pt_origin, pt_direction);
        let bvh = BVHNode::build(&obj, (0..obj.triangles.len()).collect(), 0);
        for tri in obj.triangles.iter() {
            if let Some(t) = tri.intersect(&ray, &obj.vertices) {
                println!("---------->{t}");
            }
        }
        // Perform intersection test on bvh.
        if let Some((t, _vert)) = bvh.intersect(&obj, &ray) {
            //  _triangle is the ref to intersected triangle geometry.
            // *_triangle.intersect(&ray) (for refinements...)
            println!("Hit triangle at t = {}!", t);
            assert_eq!("1.545", format!("{t:0.3}").as_str());
        } else {
            println!("No intersection. {:?} {:?}", obj.vertices, ray);
            assert!(false);
        }
        assert!(true);
    }
}

/*
* notes:
*     deltaTime = now - start "time par frame" (this keep runtime granularity out of the equation as much as possible).
*     for realtime animation:
*     Vector3d velocity += (Vector3d accelleration * deltaTime) // ifluance velocity over time
*     Point3d position += (Vector3d Velocity * deltaTime) // this is in space unit/frame
*
*     - Accelleration with negative value can simulate gravity. (it's incrementally added to
*       velocity vector over time with a Z negative value which flip smoothlly the polarity of the
*       velocity vector.)
*
*
* notes: Fundemental for understanding basic of matrix rotation.
*        -------------------------------------------------------
*        - it would be crazy to study further without
*          a very basic understanding of fundementals.
*        - it's important to have a 'solid definition' of your understanding
*          in face of such amazing and fundemental complex system.
*          so describing formula with verbose is the only way to constraint my
*          mind on the basis.
*        -------------------------------------------------------
*     - Trigger Action matrix Entity:
*          x' = x cos(theta) - y sin(theta), y' = x sin(theta) + y cos(theta),
*
*  Description:
*      x' = x component of a rotated vector2d(x,y) from basis vectors system.
*      x cos(theta) = component x on basis system * (time) cos(theta) -> equal to 1 if cos(0.0deg)
*      y sin(theta) = component y on basis system * (time) sin(theta) -> equal to 0 if sin(0.0deg)
*      x cos(theta) - y sin(theta) = (1 * 1) - (0 * 0) if theta = 0.0 deg
*      --------------------------------------------------------------------
*      y' = y component of a rotated vector2d(x,y) from basis vectors system.
*      x sin(theta) =  component x on basis system * (time) sin(theta) -> equal to 0 if sin(0.0deg)
*      y cos(theta) =  component y on basis system * (time) cos(theta) -> equal to 0 if cos(0.0deg)
*      x sin(theta) + y cos(theta) = (1 * 0) - (0 * 1) if theta = 0.0 deg
*
*
*  this describe mathematically the rotation of an unit vector on a orthogonal basis sytem.
*  - core mechanisum cos(theta) and sin(theta) will serve to divide unit basis axis length
*  by multiply the basix length of reference (x or y) by a number from 0.0 to 1.0 giving
*  a division of (x,y) component of the rotated vector.
*
*  -   we can also simplify this concept to the fact that a dot product of a unit vector by
*      the cos(theta) or sin(theta) produce it's projection on the base vector system
*      when vector start to rotate one part of the projection length for x ar y axis or
*      (their remider) is added or substracted on the oposit vector component
*      based on which axis we are operating on this discribe the rotation of the vector.
*
*  This produce: the 4x4 matrix rotation. where you distribute a Vector3d
*  disposed in colown to each menber of the row matrix  and add hem up
*  to produce the new rotated Vector3d.
*
*  the rotation identity:
*  let rotation_z = [
               [z_rad.cos(), -z_rad.sin(), 0.0, 0.0],
               [z_rad.sin(), z_rad.cos(), 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0],
           ];

           // Rotation matrix around the X-axis
           let rotation_x = [
               [1.0, 0.0, 0.0, 0.0],
               [0.0, x_rad.cos(), -x_rad.sin(), 0.0],
               [0.0, x_rad.sin(), x_rad.cos(), 0.0],
               [0.0, 0.0, 0.0, 1.0],
           ];

           // Rotation matrix around the Y-axis
           let rotation_y = [
               [y_rad.cos(), 0.0, y_rad.sin(), 0.0],
               [0.0, 1.0, 0.0, 0.0],
               [-y_rad.sin(), 0.0, y_rad.cos(), 0.0],
               [0.0, 0.0, 0.0, 1.0],
           ];

   if someone one day read my study about transformation matrix:

   - it's dificult to warp your mind around without a step by step
     process but it's worth it to understand how this logic work
     at this end it's not so complex it's just that the step by step
     process lie throuhg a non negligable opaque abstraction if you don't
     involve yourself on the basis for a moment.
   - many are passioned by this., if you don't i guess you are on wrong place.
     so you may leave that topic with in mind that's the base of what at
     my sens we can call computing... either with a calculator a computer...
     so if you are programer it's just the leverage that you may give at your tools
     to amplify your computing capability.
     it's at the core of pation for "computing computers" it up to you
     to cheat with that or not.
*
*/
