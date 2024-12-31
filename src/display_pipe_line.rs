pub mod redering_object {
    // Vertex is use as multi purpose usage
    // can be use as Point3d or Vector3d.
    // mainly designed for the display pipe line
    // and remove ambiguity when scaled to fit the display
    // unit system.
    #[derive(Debug, Copy, Clone, PartialOrd)]
    pub struct Vertex {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }
    use crate::rust3d::geometry::{Point3d, Vector3d};
    use std::hash::{Hash, Hasher};
    use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
    impl Sub for Vertex {
        type Output = Self; // Specify the result type of the addition
        fn sub(self, other: Self) -> Self {
            Vertex::new(self.x - other.x, self.y - other.y, self.z - other.z)
        }
    }
    impl Add for Vertex {
        type Output = Self; // Specify the result type of the addition
        fn add(self, other: Self) -> Self {
            Vertex::new(self.x + other.x, self.y + other.y, self.z + other.z)
        }
    }
    impl Mul for Vertex {
        type Output = Self; // Specify the result type of the addition
        fn mul(self, other: Self) -> Self {
            Vertex::new(self.x * other.x, self.y * other.y, self.z * other.z)
        }
    }
    impl PartialEq for Vertex {
        fn eq(&self, other: &Self) -> bool {
            self.x == other.x && self.y == other.y && self.z == other.z
        }
    }
    impl Eq for Vertex {}
    use std::fmt;
    impl fmt::Display for Vertex {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "Vertex(x: {:.2}, y: {:.2}, z: {:.2})",
                self.x, self.y, self.z
            )
        }
    }
    impl AddAssign<Vertex> for Vertex {
        fn add_assign(&mut self, other: Vertex) {
            self.x = self.x + other.x;
            self.y = self.y + other.y;
            self.z = self.z + other.z;
        }
    }

    impl MulAssign<f64> for Vertex {
        fn mul_assign(&mut self, scalar: f64) {
            self.x *= scalar;
            self.y *= scalar;
            self.z *= scalar;
        }
    }
    impl Hash for Vertex {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.x.to_bits().hash(state);
            self.y.to_bits().hash(state);
            self.z.to_bits().hash(state);
        }
    }
    impl Mul<f64> for Vertex {
        type Output = Vertex;
        fn mul(self, scalar: f64) -> Self {
            let v_x = self.x * scalar;
            let v_y = self.y * scalar;
            let v_z = self.z * scalar;
            Vertex::new(v_x, v_y, v_z)
        }
    }
    impl Vertex {
        pub fn new(x: f64, y: f64, z: f64) -> Self {
            Self { x, y, z }
        }

        pub fn cross_product(self, other: &Vertex) -> Self {
            Self {
                x: self.y * (*other).z - self.z * (*other).y,
                y: self.z * (*other).x - self.x * (*other).z,
                z: self.x * (*other).y - self.y * (*other).x,
            }
        }

        // Add two vertices
        pub fn add(&self, other: &Vertex) -> Vertex {
            Vertex {
                x: self.x + other.x,
                y: self.y + other.y,
                z: self.z + other.z,
            }
        }

        // Sub two vertices
        pub fn sub(&self, other: &Vertex) -> Vertex {
            Vertex {
                x: self.x - other.x,
                y: self.y - other.y,
                z: self.z - other.z,
            }
        }

        // Multiply a vertex by a scalar
        pub fn mul(&self, scalar: f64) -> Vertex {
            Vertex {
                x: self.x * scalar,
                y: self.y * scalar,
                z: self.z * scalar,
            }
        }

        // Divide a vertex by a scalar
        pub fn div(&self, scalar: f64) -> Vertex {
            Vertex {
                x: self.x / scalar,
                y: self.y / scalar,
                z: self.z / scalar,
            }
        }

        // Access the coordinate by axis index (0 = x, 1 = y, 2 = z)
        pub fn get_by_axis(&self, axis: usize) -> f64 {
            match axis {
                0 => self.x,
                1 => self.y,
                2 => self.z,
                _ => panic!("Invalid axis index!"), // You can handle this more gracefully if needed
            }
        }

        /// Compute the magnitude (length) of the vector
        pub fn magnitude(&self) -> f64 {
            (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
        }

        /// Normalize the vector (make it unit-length)
        pub fn unitize(&self) -> Vertex {
            let magnitude = self.magnitude();
            if magnitude == 0.0 {
                panic!("Cannot normalize a zero-length vector");
            }
            Vertex {
                x: self.x / magnitude,
                y: self.y / magnitude,
                z: self.z / magnitude,
            }
        }
        /// Test if a point is inside a given mesh.
        pub fn is_inside_a_mesh(&self, mesh_to_test: &Mesh) -> bool {
            let direction = Vertex::new(1.0, 0.0, 0.0);
            let ray = Ray::new(*self, direction);
            let mut intersection_count = 0;

            // Use a HashSet to store truncated distances as integers for hashing
            use std::collections::HashSet;
            let mut distances = HashSet::new();

            for triangle in mesh_to_test.triangles.iter() {
                if let Some(t) = triangle.intersect(&ray) {
                    // Scale and truncate to an integer to make it hashable
                    let scaled_t = (t * 10000.0).trunc() as i64;

                    // Only count the intersection if it's unique
                    if distances.insert(scaled_t) {
                        intersection_count += 1;
                    }
                }
            }
            // Odd intersection count means the point is inside the mesh
            (intersection_count % 2) != 0
        }
        pub fn to_point3d(&self) -> Point3d {
            Point3d::new(self.x, self.y, self.z)
        }
        pub fn to_vector3d(&self) -> Vector3d {
            Vector3d::new(self.x, self.y, self.z)
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub struct Triangle {
        pub v0: Vertex,
        pub v1: Vertex,
        pub v2: Vertex,
        pub normal: Vertex, // Precomputed normal vector
        pub id: u64,
    }

    impl Triangle {
        pub fn new(v0: Vertex, v1: Vertex, v2: Vertex) -> Self {
            // Represent edge 1 by a vector.
            let edge1 = Vertex {
                x: v1.x - v0.x,
                y: v1.y - v0.y,
                z: v1.z - v0.z,
            };
            // Make Vector of edge 2.
            let edge2 = Vertex {
                x: v2.x - v0.x,
                y: v2.y - v0.y,
                z: v2.z - v0.z,
            };
            // Normal is simply the cross Product of edge 1 and 2.
            let normal = Vertex::new(
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x,
            );
            //.unitize();
            Self {
                v0,
                v1,
                v2,
                normal,
                id: 0u64,
            }
        }

        /// Update triangle normals
        /// - when Mesh is scaled in a non uniform way.
        /// - when Mesh is rotated.
        pub fn recompute_triangle_normal(&mut self) {
            // Represent edge 1 by a vector.
            let edge1 = Vertex {
                x: self.v1.x - self.v0.x,
                y: self.v1.y - self.v0.y,
                z: self.v1.z - self.v0.z,
            };

            // Make Vector of edge 2.
            let edge2 = Vertex {
                x: self.v2.x - self.v0.x,
                y: self.v2.y - self.v0.y,
                z: self.v2.z - self.v0.z,
            };

            self.normal = Vertex::new(
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x,
            );
        }

        // Constructor with precomputed normal
        pub fn with_normal(v0: Vertex, v1: Vertex, v2: Vertex, normal: Vertex) -> Self {
            Self {
                v0,
                v1,
                v2,
                normal,
                id: 0,
            }
        }

        /// Compute the area of the triangle.
        pub fn get_triangle_area(&self) -> f64 {
            let va = self.v1.sub(self.v0);
            let vb = self.v2.sub(self.v0);
            va.cross_product(&vb).magnitude() / 2.0
        }

        // Compute the signed volume of the tetrahedron formed by the triangle and the origin
        pub fn signed_volume(&self) -> f64 {
            (self.v0.x * (self.v1.y * self.v2.z - self.v1.z * self.v2.y)
                - self.v0.y * (self.v1.x * self.v2.z - self.v1.z * self.v2.x)
                + self.v0.z * (self.v1.x * self.v2.y - self.v1.y * self.v2.x))
                / 6.0
        }

        // Möller–Trumbore algorithm.
        pub fn intersect(&self, ray: &Ray) -> Option<f64> {
            let edge1 = Vertex {
                x: self.v1.x - self.v0.x,
                y: self.v1.y - self.v0.y,
                z: self.v1.z - self.v0.z,
            };
            let edge2 = Vertex {
                x: self.v2.x - self.v0.x,
                y: self.v2.y - self.v0.y,
                z: self.v2.z - self.v0.z,
            };

            let h = Vertex {
                x: ray.direction.y * edge2.z - ray.direction.z * edge2.y,
                y: ray.direction.z * edge2.x - ray.direction.x * edge2.z,
                z: ray.direction.x * edge2.y - ray.direction.y * edge2.x,
            };
            let a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

            if a > -1e-8 && a < 1e-8 {
                return None; // Ray is parallel to the triangle.
            }

            let f = 1.0 / a;
            let s = Vertex {
                x: ray.origin.x - self.v0.x,
                y: ray.origin.y - self.v0.y,
                z: ray.origin.z - self.v0.z,
            };
            let u = f * (s.x * h.x + s.y * h.y + s.z * h.z);

            if u < 0.0 || u > 1.0 {
                return None;
            }

            let q = Vertex {
                x: s.y * edge1.z - s.z * edge1.y,
                y: s.z * edge1.x - s.x * edge1.z,
                z: s.x * edge1.y - s.y * edge1.x,
            };
            let v = f * (ray.direction.x * q.x + ray.direction.y * q.y + ray.direction.z * q.z);

            if v < 0.0 || u + v > 1.0 {
                return None;
            }

            let t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

            if t > 1e-8 {
                Some(t) // Intersection distance
            } else {
                None
            }
        }

        // Compute the centroid of the triangle
        pub fn center(&self) -> [f64; 3] {
            let centroid = self.v0.add(self.v1).add(self.v2).div(3.0);
            [centroid.x, centroid.y, centroid.z]
        }

        // Compute the bounding box of the triangle
        pub fn bounding_box(&self) -> AABB {
            AABB {
                min: Vertex {
                    x: self.v0.x.min(self.v1.x).min(self.v2.x),
                    y: self.v0.y.min(self.v1.y).min(self.v2.y),
                    z: self.v0.z.min(self.v1.z).min(self.v2.z),
                },
                max: Vertex {
                    x: self.v0.x.max(self.v1.x).max(self.v2.x),
                    y: self.v0.y.max(self.v1.y).max(self.v2.y),
                    z: self.v0.z.max(self.v1.z).max(self.v2.z),
                },
            }
        }
        pub fn edges(&self) -> [(Vertex, Vertex); 3] {
            [(self.v0, self.v1), (self.v1, self.v2), (self.v2, self.v0)]
        }
    }

    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{self, BufRead, BufReader, BufWriter, Write};
    use std::iter::Iterator;
    use std::sync::Arc;
    use std::sync::Mutex;
    use tobj;

    #[derive(Debug)]
    pub struct Mesh {
        pub vertices: Vec<Vertex>,
        pub triangles: Vec<Triangle>,
    }

    impl Mesh {
        pub fn new(vertices: Vec<Vertex>, triangles: Vec<Triangle>) -> Self {
            Self {
                vertices,
                triangles,
            }
        }

        /// Update the whole mesh triangles normals.
        /// - when scaled in a non uniform way.
        /// - when rotated.
        /// - this is multitreaded.
        pub fn update_mesh_normals(&mut self) {
            self.triangles
                .par_iter_mut()
                .for_each(|tri| tri.recompute_triangle_normal());
        }

        /// Test if a mesh is closed (watertight).
        pub fn is_watertight(&self) -> bool {
            let mut edge_count: HashMap<(Vertex, Vertex), i32> = HashMap::new();
            for triangle in self.triangles.iter() {
                for &(start, end) in &triangle.edges() {
                    let edge = if start < end {
                        (start, end)
                    } else {
                        (end, start)
                    };
                    let counter = edge_count.entry(edge).or_insert(0);
                    *counter += 1;
                }
            }
            edge_count.values().all(|&count| count == 2)
        }

        /// Test if a mesh is closed (watertight) using parallel processing.
        pub fn is_watertight_par(&self) -> bool {
            let edge_count = Mutex::new(HashMap::new());

            // Process triangles in parallel
            self.triangles.par_iter().for_each(|triangle| {
                for &(start, end) in &triangle.edges() {
                    let edge = if start < end {
                        (start, end)
                    } else {
                        (end, start)
                    };

                    // Lock the mutex, update the edge count, and immediately release the lock
                    let mut map = edge_count.lock().unwrap();
                    *map.entry(edge).or_insert(0) += 1;
                }
            });

            // Perform the final watertight check
            let final_map = edge_count.lock().unwrap(); // Lock again to check
            final_map.values().all(|&count| count == 2)
        }

        /// Give the volume of the mesh in file system base unit.
        /// Based on divergence theorem (also known as Gauss's theorem)
        pub fn compute_volume(&self) -> f64 {
            self.triangles.iter().map(|tri| tri.signed_volume()).sum()
        }

        /// Display Statistics of obj file.
        /// # Arguments
        ///     file path of the obj file.
        /// # Returns
        ///     Result(vertex count,normal count,face count)
        pub fn count_obj_elements(file_path: &str) -> io::Result<(usize, usize, usize)> {
            let file = File::open(file_path)?;
            let reader = BufReader::new(file);

            let mut vertex_count = 0;
            let mut normal_count = 0;
            let mut face_count = 0;

            for line in reader.lines() {
                let line = line?;
                let mut words = line.split_whitespace();

                if let Some(prefix) = words.next() {
                    match prefix {
                        "v" => vertex_count += 1,  // Vertex position
                        "vn" => normal_count += 1, // Vertex normal
                        "f" => face_count += 1,    // Face
                        _ => {}                    // Ignore other lines
                    }
                }
            }

            Ok((vertex_count, normal_count, face_count))
        }

        /// Export the mesh to an .obj file.
        pub fn export_to_obj(&self, file_path: &str) -> io::Result<()> {
            let mut file = File::create(file_path)?;

            // Write vertices
            for vertex in &self.vertices {
                writeln!(file, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
            }

            // Write faces
            for triangle in &self.triangles {
                let v0_idx = self
                    .vertices
                    .iter()
                    .position(|v| *v == triangle.v0)
                    .unwrap()
                    + 1;
                let v1_idx = self
                    .vertices
                    .iter()
                    .position(|v| *v == triangle.v1)
                    .unwrap()
                    + 1;
                let v2_idx = self
                    .vertices
                    .iter()
                    .position(|v| *v == triangle.v2)
                    .unwrap()
                    + 1;

                writeln!(file, "f {} {} {}", v0_idx, v1_idx, v2_idx)?;
            }
            Ok(())
        }

        /// Import a mesh from an .obj file Only work with triangles.
        pub fn import_from_obj_tri_only(
            file_path: &str,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let (models, _materials) = tobj::load_obj(file_path, &tobj::LoadOptions::default())?;
            let mut vertices = Vec::new();
            let mut triangles = Vec::new();

            for model in models {
                let mesh = model.mesh;

                // Add vertices
                for i in 0..mesh.positions.len() / 3 {
                    vertices.push(Vertex {
                        x: mesh.positions[3 * i] as f64,
                        y: mesh.positions[3 * i + 1] as f64,
                        z: mesh.positions[3 * i + 2] as f64,
                    });
                }

                // Add triangles
                for i in 0..mesh.indices.len() / 3 {
                    let v0 = vertices[mesh.indices[3 * i] as usize];
                    let v1 = vertices[mesh.indices[3 * i + 1] as usize];
                    let v2 = vertices[mesh.indices[3 * i + 2] as usize];

                    triangles.push(Triangle::new(v0, v1, v2));
                }
            }
            Ok(Self {
                vertices,
                triangles,
            })
        }

        pub fn import_from_obj(file_path: &str) -> io::Result<Self> {
            let file = File::open(file_path)?;
            let reader = io::BufReader::new(file);

            let mut vertices: Vec<Vertex> = Vec::new();
            let mut triangles: Vec<Triangle> = Vec::new();

            for line in reader.lines() {
                let line = line?;
                let parts: Vec<&str> = line.split_whitespace().collect();

                if parts.is_empty() {
                    continue; // Skip empty lines
                }

                match parts[0] {
                    "v" => {
                        // Parse a vertex
                        let x: f64 = parts[1].parse().expect("Invalid vertex format");
                        let y: f64 = parts[2].parse().expect("Invalid vertex format");
                        let z: f64 = parts[3].parse().expect("Invalid vertex format");
                        vertices.push(Vertex::new(x, y, z));
                    }
                    "f" => {
                        // Parse a face (convert quads to triangles)
                        let indices: Vec<usize> = parts[1..]
                            .iter()
                            .map(|s| s.split('/').next().unwrap().parse::<usize>().unwrap() - 1)
                            .collect();

                        if indices.len() == 3 {
                            // Triangle face
                            triangles.push(Triangle::new(
                                vertices[indices[0]],
                                vertices[indices[1]],
                                vertices[indices[2]],
                            ));
                        } else if indices.len() == 4 {
                            // Quad face, split into two triangles
                            triangles.push(Triangle::new(
                                vertices[indices[0]],
                                vertices[indices[1]],
                                vertices[indices[2]],
                            ));
                            triangles.push(Triangle::new(
                                vertices[indices[0]],
                                vertices[indices[2]],
                                vertices[indices[3]],
                            ));
                        } else {
                            panic!("Unsupported face format: more than 4 vertices per face.");
                        }
                    }
                    _ => {
                        // Ignore other lines
                    }
                }
            }
            Ok(Mesh::new(vertices, triangles))
        }

        /// an efficient way to import obj files.
        pub fn import_obj_with_normals(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
            use std::fs::File;
            use std::io::{BufRead, BufReader};

            let file = File::open(path)?;
            let reader = BufReader::new(file);

            let mut vertices = Vec::new();
            let mut normals = Vec::new();
            let mut triangles = Vec::new();

            for line in reader.lines() {
                let line = line?;
                let parts: Vec<&str> = line.split_whitespace().collect();

                if parts.is_empty() {
                    continue;
                }

                match parts[0] {
                    "v" => {
                        // Vertex position
                        let x: f64 = parts[1].parse()?;
                        let y: f64 = parts[2].parse()?;
                        let z: f64 = parts[3].parse()?;
                        vertices.push(Vertex::new(x, y, z));
                    }
                    "vn" => {
                        // Vertex normal
                        let x: f64 = parts[1].parse()?;
                        let y: f64 = parts[2].parse()?;
                        let z: f64 = parts[3].parse()?;
                        normals.push(Vertex::new(x, y, z).unitize());
                    }
                    "f" => {
                        // Face
                        let mut face_vertices = Vec::new();
                        let mut face_normals = Vec::new();

                        for part in &parts[1..] {
                            let indices: Vec<&str> = part.split('/').collect();
                            let vertex_idx: usize = indices[0].parse::<usize>()? - 1; // .obj is 1-indexed
                            face_vertices.push(vertices[vertex_idx]);

                            // If normals are available
                            if indices.len() > 2 && !indices[2].is_empty() {
                                let normal_idx: usize = indices[2].parse::<usize>()? - 1;
                                face_normals.push(normals[normal_idx]);
                            }
                        }

                        if face_vertices.len() == 3 {
                            if face_normals.len() == 3 {
                                // Use the first normal for the entire triangle
                                triangles.push(Triangle::with_normal(
                                    face_vertices[0],
                                    face_vertices[1],
                                    face_vertices[2],
                                    face_normals[0],
                                ));
                            } else {
                                // Compute normal if not provided
                                triangles.push(Triangle::new(
                                    face_vertices[0],
                                    face_vertices[1],
                                    face_vertices[2],
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Mesh {
                vertices,
                triangles,
            })
        }

        pub fn export_to_obj_with_normals(
            &self,
            path: &str,
        ) -> Result<(), Box<dyn std::error::Error>> {
            use std::fs::File;
            use std::io::{BufWriter, Write};

            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);
            println!("\x1b[2J");
            println!("\x1b[3;0HExporting (speed not optimized) Path:({0})", path);
            let vertex_count = self.vertices.len();
            let mut ct = 0;
            for vertex in &self.vertices {
                writeln!(writer, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
                ct += 1;
                println!("\x1b[4;0HVertex step Progress:{0}/{1}", ct, vertex_count);
            }

            let triangle_count = self.triangles.len();
            ct = 0;
            for triangle in &self.triangles {
                writeln!(
                    writer,
                    "vn {} {} {}",
                    triangle.normal.x, triangle.normal.y, triangle.normal.z
                )?;
                ct += 1;
                println!(
                    "\x1b[5;0HVertex Normals step Progress:{0}/{1}",
                    ct, triangle_count
                );
            }
            ct = 0;
            for triangle in &self.triangles {
                let v0_idx = self
                    .vertices
                    .iter()
                    .position(|v| *v == triangle.v0)
                    .unwrap()
                    + 1;
                let v1_idx = self
                    .vertices
                    .iter()
                    .position(|v| *v == triangle.v1)
                    .unwrap()
                    + 1;
                let v2_idx = self
                    .vertices
                    .iter()
                    .position(|v| *v == triangle.v2)
                    .unwrap()
                    + 1;
                writeln!(
                    writer,
                    "f {}/{} {}/{} {}/{}",
                    v0_idx, v0_idx, v1_idx, v1_idx, v2_idx, v2_idx
                )?;
                ct += 1;
                println!("\x1b[6;0HFace(s) step Progress:{0}/{1}", ct, triangle_count);
            }
            Ok(())
        }

        pub fn export_to_obj_with_normals_fast(
            &self,
            path: &str,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);

            println!("\x1b[2J");
            println!("\x1b[3;0HExporting (optimized) Path:({})", path);

            // Step 1: Precompute vertex-to-index mapping
            let vertex_index_map: HashMap<&Vertex, usize> = self
                .vertices
                .iter()
                .enumerate()
                .map(|(i, v)| (v, i + 1)) // OBJ indices are 1-based
                .collect();

            let vertex_count = self.vertices.len();
            let triangle_count = self.triangles.len();

            // Step 2: Generate vertex data in parallel
            println!("\x1b[4;0HGenerating vertex data...");
            let vertex_data: Vec<String> = self
                .vertices
                .par_iter()
                .map(|vertex| format!("v {} {} {}\n", vertex.x, vertex.y, vertex.z))
                .collect();

            // Step 3: Generate normal data in parallel
            println!("\x1b[5;0HGenerating normal data...");
            let normal_data: Vec<String> = self
                .triangles
                .par_iter()
                .map(|triangle| {
                    format!(
                        "vn {} {} {}\n",
                        triangle.normal.x, triangle.normal.y, triangle.normal.z
                    )
                })
                .collect();

            // Step 4: Generate face data in parallel
            println!("\x1b[6;0HGenerating face data...");
            let face_data: Vec<String> = self
                .triangles
                .par_iter()
                .map(|triangle| {
                    let v0_idx = *vertex_index_map.get(&triangle.v0).unwrap();
                    let v1_idx = *vertex_index_map.get(&triangle.v1).unwrap();
                    let v2_idx = *vertex_index_map.get(&triangle.v2).unwrap();
                    format!(
                        "f {}/{} {}/{} {}/{}\n",
                        v0_idx, v0_idx, v1_idx, v1_idx, v2_idx, v2_idx
                    )
                })
                .collect();

            // Step 5: Write data to file in batches
            println!("\x1b[7;0HWriting data to file...");
            for (i, line) in vertex_data.iter().enumerate() {
                writer.write_all(line.as_bytes())?;
                if i % 1000 == 0 || i == vertex_count - 1 {
                    println!("\x1b[4;0HVertex step Progress: {}/{}", i + 1, vertex_count);
                }
            }

            for (i, line) in normal_data.iter().enumerate() {
                writer.write_all(line.as_bytes())?;
                if i % 1000 == 0 || i == triangle_count - 1 {
                    println!(
                        "\x1b[5;0HVertex Normals step Progress: {}/{}",
                        i + 1,
                        triangle_count
                    );
                }
            }

            for (i, line) in face_data.iter().enumerate() {
                writer.write_all(line.as_bytes())?;
                if i % 1000 == 0 || i == triangle_count - 1 {
                    println!(
                        "\x1b[6;0HFace(s) step Progress: {}/{}",
                        i + 1,
                        triangle_count
                    );
                }
            }

            println!("\x1b[8;0HExport completed successfully!");
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub struct Ray {
        pub origin: Vertex,
        pub direction: Vertex,
    }

    impl Ray {
        pub fn new(pt1: Vertex, pt2: Vertex) -> Self {
            Self {
                origin: pt1,
                direction: pt2,
            }
        }
    }

    // A Bounding Volume Hierarchy (BVH) organizes objects (e.g., triangles)
    // into a tree structure to accelerate ray tracing
    // by reducing the number of intersection tests.

    // AABB stand for Axis aligned Bounding Box.
    #[derive(Debug, Clone)]
    pub struct AABB {
        min: Vertex, // Minimum corner of the bounding box
        max: Vertex, // Maximum corner of the bounding box
    }

    impl AABB {
        // Combine two AABBs into one that encompasses both
        pub fn surrounding_box(box1: &AABB, box2: &AABB) -> AABB {
            AABB {
                min: Vertex {
                    x: box1.min.x.min(box2.min.x),
                    y: box1.min.y.min(box2.min.y),
                    z: box1.min.z.min(box2.min.z),
                },
                max: Vertex {
                    x: box1.max.x.max(box2.max.x),
                    y: box1.max.y.max(box2.max.y),
                    z: box1.max.z.max(box2.max.z),
                },
            }
        }

        // Ray-AABB intersection test (needed for BVH traversal)
        pub fn intersects(&self, ray: &Ray) -> bool {
            let inv_dir = Vertex {
                x: 1.0 / ray.direction.x,
                y: 1.0 / ray.direction.y,
                z: 1.0 / ray.direction.z,
            };

            let t_min = (
                (self.min.x - ray.origin.x) * inv_dir.x,
                (self.min.y - ray.origin.y) * inv_dir.y,
                (self.min.z - ray.origin.z) * inv_dir.z,
            );
            let t_max = (
                (self.max.x - ray.origin.x) * inv_dir.x,
                (self.max.y - ray.origin.y) * inv_dir.y,
                (self.max.z - ray.origin.z) * inv_dir.z,
            );

            let t_enter = t_min.0.max(t_min.1).max(t_min.2);
            let t_exit = t_max.0.min(t_max.1).min(t_max.2);

            t_enter <= t_exit
        }
    }

    #[derive(Debug)]
    pub enum BVHNode {
        Leaf {
            bounding_box: AABB,
            triangles: Vec<Triangle>, // Triangles in the leaf
        },
        Internal {
            bounding_box: AABB,
            left: Arc<BVHNode>,  // Left child
            right: Arc<BVHNode>, // Right child
        },
    }

    impl BVHNode {
        // A Bounding Volume Hierarchy (BVH) organizes objects (e.g., triangles)
        // into a tree structure to accelerate ray tracing by
        // reducing the number of intersection tests.
        pub fn build(triangles: Vec<Triangle>, depth: usize) -> BVHNode {
            // Base case: Create a leaf node if triangle count is small
            if triangles.len() <= 2 {
                let bounding_box = triangles
                    .iter()
                    .map(|tri| tri.bounding_box())
                    .reduce(|a, b| AABB::surrounding_box(&a, &b))
                    .unwrap();
                return BVHNode::Leaf {
                    bounding_box,
                    triangles,
                };
            }

            // Find the axis to split (X, Y, or Z)
            let axis = depth % 3;
            let mut sorted_triangles = triangles;
            sorted_triangles.sort_by(|a, b| {
                let center_a = a.center()[axis];
                let center_b = b.center()[axis];
                center_a.partial_cmp(&center_b).unwrap()
            });

            // Partition the triangles into two groups
            let mid = sorted_triangles.len() / 2;
            let (left_triangles, right_triangles) = sorted_triangles.split_at(mid);

            // Recursively build the left and right subtrees
            let left = Arc::new(BVHNode::build(left_triangles.to_vec(), depth + 1));
            let right = Arc::new(BVHNode::build(right_triangles.to_vec(), depth + 1));

            // Create the bounding box for this node
            let bounding_box = AABB::surrounding_box(&left.bounding_box(), &right.bounding_box());

            BVHNode::Internal {
                bounding_box,
                left,
                right,
            }
        }

        pub fn bounding_box(&self) -> &AABB {
            match self {
                BVHNode::Leaf { bounding_box, .. } => bounding_box,
                BVHNode::Internal { bounding_box, .. } => bounding_box,
            }
        }
        pub fn intersect(&self, ray: &Ray) -> Option<(f64, &Triangle)> {
            if !self.bounding_box().intersects(ray) {
                return None; // Ray doesn't hit this node
            }

            match self {
                BVHNode::Leaf { triangles, .. } => {
                    // Test against all triangles in the leaf
                    let mut closest_hit: Option<(f64, &Triangle)> = None;
                    for triangle in triangles {
                        if let Some(t) = triangle.intersect(ray) {
                            if closest_hit.is_none() || t < closest_hit.unwrap().0 {
                                closest_hit = Some((t, triangle));
                            }
                        }
                    }
                    closest_hit
                }
                BVHNode::Internal { left, right, .. } => {
                    // Recursively test left and right children
                    let left_hit = left.intersect(ray);
                    let right_hit = right.intersect(ray);

                    match (left_hit, right_hit) {
                        (Some(l), Some(r)) => {
                            if l.0 < r.0 {
                                Some(l)
                            } else {
                                Some(r)
                            }
                        }
                        (Some(l), None) => Some(l),
                        (None, Some(r)) => Some(r),
                        (None, None) => None,
                    }
                }
            }
        }
    }

    pub struct Camera {
        position: Vertex,  // Camera position
        forward: Vertex,   // Forward direction
        right: Vertex,     // Right direction
        up: Vertex,        // Up direction
        fov: f64,          // Field of view in degrees
        aspect_ratio: f64, // Aspect ratio (width / height)
        width: usize,      // Image width
        height: usize,     // Image height
    }

    impl Camera {
        // Generate rays for a given pixel
        pub fn generate_ray(&self, pixel_x: usize, pixel_y: usize) -> Ray {
            // Convert FOV to radians and compute scaling factor
            let fov_scale = (self.fov.to_radians() / 2.0).tan();

            // Map pixel to normalized device coordinates (NDC)
            let ndc_x = (pixel_x as f64 + 0.5) / self.width as f64; // Center pixel
            let ndc_y = (pixel_y as f64 + 0.5) / self.height as f64;

            // Map NDC to screen space [-1, 1]
            let screen_x = 2.0 * ndc_x - 1.0;
            let screen_y = 1.0 - 2.0 * ndc_y;

            // Adjust for aspect ratio and FOV
            let pixel_camera_x = screen_x * self.aspect_ratio * fov_scale;
            let pixel_camera_y = screen_y * fov_scale;

            // Compute ray direction in camera space
            let ray_direction = self
                .forward
                .add(self.right.mul(pixel_camera_x))
                .add(self.up.mul(pixel_camera_y))
                .unitize(); // Normalize to get unit vector

            // Create the ray
            Ray {
                origin: self.position,
                direction: ray_direction,
            }
        }

        // Generate a memory allocated array: Vec<Ray> of Ray object for further
        // Ray tracing from Camera fov generated rays.
        pub fn render(camera: &Camera, ray_buffer: &mut Vec<Ray>) {
            for y in 0..camera.height {
                for x in 0..camera.width {
                    let ray = camera.generate_ray(x, y);
                    println!(
                        "Pixel ({}, {}) -> Ray Origin: {:?}, Direction: {:?}",
                        x, y, ray.origin, ray.direction
                    );
                    ray_buffer.push(ray);
                }
            }
        }
    }
    mod ray_operations {
        fn shade_with_distance(
            base_color: (u8, u8, u8),
            distance: f64,
            attenuation: f64,
        ) -> (u8, u8, u8) {
            let intensity = 1.0 / (1.0 + attenuation * distance);
            let (r, g, b) = base_color;
            (
                (r as f64 * intensity) as u8,
                (g as f64 * intensity) as u8,
                (b as f64 * intensity) as u8,
            )
        }
        fn fog_with_distance(
            base_color: (u8, u8, u8),
            fog_color: (u8, u8, u8),
            distance: f64,
            max_distance: f64,
        ) -> (u8, u8, u8) {
            let fog_factor = (distance / max_distance).min(1.0);
            let (r1, g1, b1) = base_color;
            let (r2, g2, b2) = fog_color;

            (
                ((r1 as f64 * (1.0 - fog_factor)) + (r2 as f64 * fog_factor)) as u8,
                ((g1 as f64 * (1.0 - fog_factor)) + (g2 as f64 * fog_factor)) as u8,
                ((b1 as f64 * (1.0 - fog_factor)) + (b2 as f64 * fog_factor)) as u8,
            )
        }
        fn light_falloff(base_color: (u8, u8, u8), distance: f64) -> (u8, u8, u8) {
            let intensity = 1.0 / (distance + 1.0); // Soft falloff
            let (r, g, b) = base_color;

            (
                (r as f64 * intensity) as u8,
                (g as f64 * intensity) as u8,
                (b as f64 * intensity) as u8,
            )
        }
    }
}
/*
 *     Visualization V3 bring Vertex standardization
 *     and some code optimization providing matrix 4x4
 *     4x3 operations for Vertex translation and rotation.
 *     - perspective projection are always in 4x4
 *       (with Homogeneous transformation)
 *     - apply transformation matrix in parallel,
 *       on method with suffix _par
 */
pub mod visualization_v3 {
    use super::redering_object::Vertex;
    use crate::rust3d::geometry::{Point3d, Vector3d};
    use rayon::prelude::*;

    pub struct Camera {
        pub position: Point3d,
        pub target: Point3d,
        pub up: Vector3d,
        pub fov: f64,
        pub width: f64,
        pub height: f64,
        pub near: f64,
        pub far: f64,
        pub view_matrix: [[f64; 4]; 4], // Precomputed view matrix
        pub projection_matrix: [[f64; 4]; 4], // Precomputed projection matrix
    }

    impl Camera {
        /// Construct a camera with cached matrix conversion
        /// which involve to update the matrix if one of the camera component
        /// has changed. use .update_matrices() on the camera object for that
        /// purpose.
        pub fn new(
            position: Point3d,
            target: Point3d,
            up: Vector3d,
            width: f64,
            height: f64,
            fov: f64,
            near: f64,
            far: f64,
        ) -> Self {
            let mut camera = Self {
                position,
                target,
                up,
                fov,
                width,
                height,
                near,
                far,
                view_matrix: [[0.0; 4]; 4], // Temporary initialization
                projection_matrix: [[0.0; 4]; 4], // Temporary initialization
            };

            // Precompute the matrices
            camera.update_matrices();
            camera
        }

        /// Project a set of Vertex on 2d space.
        /// this is multitreaded.
        /// # Returns
        /// a vector of tuple (x,y,depth) (usize,usize,f64)
        pub fn project_points(&self, points: &Vec<Vertex>) -> Vec<(usize, usize, f64)> {
            points
                .par_iter() // Use parallel iterator
                .filter_map(|point| self.project(*point)) // Apply the project method in parallel
                .collect() // Collect results into a Vec
        }

        /// Update the view and projection matrices (call when camera parameters change)
        pub fn update_matrices(&mut self) {
            self.view_matrix = self.compute_view_matrix();
            self.projection_matrix = self.compute_projection_matrix();
        }

        /// Compute the view matrix
        pub fn compute_view_matrix(&self) -> [[f64; 4]; 4] {
            let forward = Vector3d::new(
                self.position.X - self.target.X,
                self.position.Y - self.target.Y,
                self.position.Z - self.target.Z,
            )
            .unitize_b();
            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Vertex as Vector3d to avoid run time penalty.
            let translation = Vertex::new(-self.position.X, -self.position.Y, -self.position.Z);
            [
                [right.get_X(), up.get_X(), -forward.get_X(), 0.0],
                [right.get_Y(), up.get_Y(), -forward.get_Y(), 0.0],
                [right.get_Z(), up.get_Z(), -forward.get_Z(), 0.0],
                [
                    (right.get_X() * translation.x)
                        + (right.get_Y() * translation.y)
                        + (right.get_Z() * translation.z),
                    (up.get_X() * translation.x)
                        + (up.get_Y() * translation.y)
                        + (right.get_Z() * translation.z),
                    (forward.get_X() * translation.x)
                        + (forward.get_Y() * translation.y)
                        + (forward.get_Z() * translation.z),
                    1.0,
                ],
            ]
        }

        /// Compute the projection matrix
        pub fn compute_projection_matrix(&self) -> [[f64; 4]; 4] {
            let aspect_ratio = self.width / self.height;
            let fov_rad = self.fov.to_radians();
            let f = 1.0 / (fov_rad / 2.0).tan();
            [
                [f / aspect_ratio, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    (self.far + self.near) / (self.near - self.far),
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    (2.0 * self.far * self.near) / (self.near - self.far),
                    0.0,
                ],
            ]
        }

        /// Project a vertex in a 2d camera space projection.
        pub fn project(&self, point: Vertex) -> Option<(usize, usize, f64)> {
            // Use precomputed matrices
            let camera_space_point = self.multiply_matrix_vector(self.view_matrix, point);

            // Extract depth in camera space (before projection)
            let depth_in_camera_space = camera_space_point.z;

            let projected_point =
                self.multiply_matrix_vector(self.projection_matrix, camera_space_point);

            // Homogeneous divide (perspective divide)
            let x = projected_point.x / projected_point.z;
            let y = projected_point.y / projected_point.z;

            // Map the coordinates from [-1, 1] to screen space
            let screen_x = ((x + 1.0) * 0.5 * self.width) as isize;
            let screen_y = ((1.0 - y) * 0.5 * self.height) as isize;

            if screen_x < 0
                || screen_x >= self.width as isize
                || screen_y < 0
                || screen_y >= self.height as isize
            {
                return None; // Point is out of screen bounds
            }

            Some((screen_x as usize, screen_y as usize, depth_in_camera_space))
        }

        pub fn multiply_matrix_vector(&self, matrix: [[f64; 4]; 4], v: Vertex) -> Vertex {
            Vertex::new(
                matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z + matrix[0][3],
                matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z + matrix[1][3],
                matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z + matrix[2][3],
            )
        }

        /// Generate a transformation matrix for the camera's movement and rotation
        pub fn get_transformation_matrix(
            &self,
            forward_amount: f64,
            yaw_angle: f64,
            pitch_angle: f64,
        ) -> [[f64; 4]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vertex::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .unitize();

            // Step 2: Compute the translation vector for the camera movement
            let translation = forward * (-forward_amount);

            // Step 3: Compute rotation matrices
            // Yaw (rotation around the Y-axis)
            let yaw_cos = yaw_angle.to_radians().cos();
            let yaw_sin = yaw_angle.to_radians().sin();
            let yaw_matrix = [
                [yaw_cos, 0.0, yaw_sin, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-yaw_sin, 0.0, yaw_cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Pitch (rotation around the X-axis)
            let pitch_cos = pitch_angle.to_radians().cos();
            let pitch_sin = pitch_angle.to_radians().sin();
            let pitch_matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, pitch_cos, -pitch_sin, 0.0],
                [0.0, pitch_sin, pitch_cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Step 4: Combine yaw and pitch into a single rotation matrix
            let rotation_matrix = self.multiply_matrices(yaw_matrix, pitch_matrix);

            // Step 5: Create the translation matrix
            let translation_matrix = [
                [1.0, 0.0, 0.0, translation.x],
                [0.0, 1.0, 0.0, translation.y],
                [0.0, 0.0, 1.0, translation.z],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Step 6: Combine rotation and translation matrices into a single transformation matrix
            self.multiply_matrices(rotation_matrix, translation_matrix)
        }

        /// Apply a transformation matrix to a Vec<Vertex> and return the transformed points
        pub fn transform_points(
            &self,
            points: &Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) -> Vec<Vertex> {
            points
                .iter()
                .map(|point| self.multiply_matrix_vector(transformation_matrix, *point))
                .collect()
        }
        /// Apply a transformation matrix to a Vec<Vertex> and return the transformed points
        /// parallelized version of the method.
        pub fn transform_points_par(
            &self,
            points: &Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) -> Vec<Vertex> {
            points
                .par_iter()
                .map(|point| self.multiply_matrix_vector(transformation_matrix, *point))
                .collect()
        }

        /// Apply a transformation matrix to a mutable Vec<Vertex>
        pub fn transform_points_mut(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) {
            points.iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector(transformation_matrix, *point);
            });
        }
        /// Apply a transformation matrix to a mutable Vec<Vertex>
        /// parallelized version of the method.
        pub fn transform_points_mut_par(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 4]; 4],
        ) {
            points.par_iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector(transformation_matrix, *point);
            });
        }
        /// Generate a transformation matrix for panning the camera
        pub fn get_pan_matrix(&self, right_amount: f64, up_amount: f64) -> [[f64; 4]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vector3d::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .unitize_b();

            // Step 2: Compute the right direction vector (cross product of forward and up)
            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();

            // Step 3: Compute the true up direction (orthogonalized)
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Step 4: Scale the right and up vectors by the panning amounts
            let pan_translation =
                right.to_vertex() * (-right_amount) + (up.to_vertex() * up_amount);

            // Step 5: Create the translation matrix for panning
            [
                [1.0, 0.0, 0.0, pan_translation.x],
                [0.0, 1.0, 0.0, pan_translation.y],
                [0.0, 0.0, 1.0, pan_translation.z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }

        /// Apply panning to a set of Vertex.
        pub fn pan_points(
            &self,
            points: &Vec<Vertex>,
            right_amount: f64,
            up_amount: f64,
        ) -> Vec<Vertex> {
            // Step 1: Get the pan transformation matrix
            let pan_matrix = self.get_pan_matrix(right_amount, up_amount);

            // Step 2: Apply the pan transformation to all points
            points
                .iter()
                .map(|point| self.multiply_matrix_vector(pan_matrix, *point))
                .collect()
        }

        /// Apply panning to a set of mutable Vertex points.
        pub fn pan_points_mut(&self, points: &mut Vec<Vertex>, right_amount: f64, up_amount: f64) {
            // Step 1: Get the pan transformation matrix
            let pan_matrix = self.get_pan_matrix(right_amount, up_amount);

            // Step 2: Apply the pan transformation to all points
            points.iter_mut().for_each(|point| {
                (*point) = self.multiply_matrix_vector(pan_matrix, *point);
            });
        }

        /// Generate a panning transformation matrix
        /// `dx` and `dy` are the offsets in world space along the right and up directions.
        pub fn pan_point_matrix(&self, dx: f64, dy: f64) -> [[f64; 4]; 4] {
            // Calculate the right and up vectors based on the camera's orientation
            let forward = Vector3d::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .unitize_b();

            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Translation in the right and up directions
            let translation = Vertex::new(
                dx * right.get_X() + dy * up.get_X(),
                dx * right.get_Y() + dy * up.get_Y(),
                dx * right.get_Z() + dy * up.get_Z(),
            );

            // Construct the transformation matrix
            [
                [1.0, 0.0, 0.0, translation.x],
                [0.0, 1.0, 0.0, translation.y],
                [0.0, 0.0, 1.0, translation.z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }

        /// Create a rotation matrix from angles (in degrees) for X, Y, and Z axes
        pub fn rotation_matrix_from_angles(
            x_angle: f64,
            y_angle: f64,
            z_angle: f64,
        ) -> [[f64; 4]; 4] {
            // Convert angles from degrees to radians
            let x_rad = x_angle.to_radians();
            let y_rad = y_angle.to_radians();
            let z_rad = z_angle.to_radians();

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

            // Rotation matrix around the Z-axis
            let rotation_z = [
                [z_rad.cos(), -z_rad.sin(), 0.0, 0.0],
                [z_rad.sin(), z_rad.cos(), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Combine the rotation matrices: R = Rz * Ry * Rx
            let rotation_xy = Self::multiply_matrices_axb(rotation_y, rotation_x);
            let rotation_xyz = Self::multiply_matrices_axb(rotation_z, rotation_xy);

            rotation_xyz
        }

        /// Combine multiple transformation matrices into one
        /// they have to be call from a stack vector use macro vec! for that.
        pub fn combine_matrices(matrices: Vec<[[f64; 4]; 4]>) -> [[f64; 4]; 4] {
            // Start with the identity matrix
            let mut result = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];

            // Multiply all matrices together in sequence
            for matrix in matrices {
                result = Self::multiply_matrices_axb(result, matrix);
            }

            result
        }

        /// Helper function to multiply two 4x4 matrices
        pub fn multiply_matrices_axb(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
            let mut result = [[0.0; 4]; 4];

            for i in 0..4 {
                for j in 0..4 {
                    result[i][j] = a[i][0] * b[0][j]
                        + a[i][1] * b[1][j]
                        + a[i][2] * b[2][j]
                        + a[i][3] * b[3][j];
                }
            }
            result
        }
        /// Utility function: Multiply two 4x4 matrices (non static method version).
        fn multiply_matrices(&self, a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
            let mut result = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    result[i][j] = a[i][0] * b[0][j]
                        + a[i][1] * b[1][j]
                        + a[i][2] * b[2][j]
                        + a[i][3] * b[3][j];
                }
            }
            result
        }
        // 4x3 matrix version of the code, the absance of the Homogeneous Normalization
        // modify the sensitivity of the system.
        // but provide a boost in computation performance
        // (from the drop of the last row of the 4x4 matrix)
        // use regular 4x4 version if you preferes more precision in motion.
        // - all 4x3 operations are explicitly named with _4x3 sufix.
        /// Generate a transformation matrix for the camera's movement and rotation
        pub fn get_transformation_matrix_4x3(
            &self,
            forward_amount: f64,
            yaw_angle: f64,
            pitch_angle: f64,
        ) -> [[f64; 3]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vertex::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .unitize();

            // Step 2: Compute the translation vector for the camera movement
            let translation = forward * (-forward_amount);

            // Step 3: Compute rotation matrices
            // Yaw (rotation around the Y-axis)
            let yaw_cos = yaw_angle.to_radians().cos();
            let yaw_sin = yaw_angle.to_radians().sin();
            let yaw_matrix = [
                [yaw_cos, 0.0, yaw_sin],
                [0.0, 1.0, 0.0],
                [-yaw_sin, 0.0, yaw_cos],
            ];

            // Pitch (rotation around the X-axis)
            let pitch_cos = pitch_angle.to_radians().cos();
            let pitch_sin = pitch_angle.to_radians().sin();
            let pitch_matrix = [
                [1.0, 0.0, 0.0],
                [0.0, pitch_cos, -pitch_sin],
                [0.0, pitch_sin, pitch_cos],
            ];

            // Step 4: Combine yaw and pitch into a single rotation matrix
            let rotation_matrix = self.multiply_matrices_3x3(yaw_matrix, pitch_matrix);

            // Step 5: Combine the rotation matrix and translation into a single 4x3 transformation matrix
            [
                [
                    rotation_matrix[0][0],
                    rotation_matrix[0][1],
                    rotation_matrix[0][2],
                ],
                [
                    rotation_matrix[1][0],
                    rotation_matrix[1][1],
                    rotation_matrix[1][2],
                ],
                [
                    rotation_matrix[2][0],
                    rotation_matrix[2][1],
                    rotation_matrix[2][2],
                ],
                [translation.x, translation.y, translation.z],
            ]
        }

        /// Apply a transformation matrix to a Vec<Vertex> and return the transformed points
        pub fn transform_points_4x3(
            &self,
            points: &Vec<Vertex>,
            transformation_matrix: [[f64; 3]; 4],
        ) -> Vec<Vertex> {
            points
                .iter()
                .map(|point| self.multiply_matrix_vector_4x3(transformation_matrix, *point))
                .collect()
        }

        /// Apply a transformation matrix to a Vec<Vertex> and return the transformed points
        /// parallelized version of the method.
        pub fn transform_points_4x3_par(
            &self,
            points: &Vec<Vertex>,
            transformation_matrix: [[f64; 3]; 4],
        ) -> Vec<Vertex> {
            points
                .par_iter()
                .map(|point| self.multiply_matrix_vector_4x3(transformation_matrix, *point))
                .collect()
        }

        /// Apply a transformation matrix to a mutable Vec<Vertex>.
        pub fn transform_points_mut_4x3(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 3]; 4],
        ) {
            points.iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector_4x3(transformation_matrix, *point);
            });
        }

        /// Apply a transformation matrix to a mutable Vec<Vertex> parallelized version.
        /// parallelized version of the method.
        pub fn transform_points_mut_4x3_par(
            &self,
            points: &mut Vec<Vertex>,
            transformation_matrix: [[f64; 3]; 4],
        ) {
            points.par_iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector_4x3(transformation_matrix, *point);
            });
        }
        pub fn rotation_matrix_from_angles_4x3(
            &self,
            x_angle: f64,
            y_angle: f64,
            z_angle: f64,
        ) -> [[f64; 3]; 4] {
            // Convert angles from degrees to radians
            let x_rad = x_angle.to_radians();
            let y_rad = y_angle.to_radians();
            let z_rad = z_angle.to_radians();

            // Rotation matrix around the X-axis
            let rotation_x = [
                [1.0, 0.0, 0.0],
                [0.0, x_rad.cos(), -x_rad.sin()],
                [0.0, x_rad.sin(), x_rad.cos()],
            ];

            // Rotation matrix around the Y-axis
            let rotation_y = [
                [y_rad.cos(), 0.0, y_rad.sin()],
                [0.0, 1.0, 0.0],
                [-y_rad.sin(), 0.0, y_rad.cos()],
            ];

            // Rotation matrix around the Z-axis
            let rotation_z = [
                [z_rad.cos(), -z_rad.sin(), 0.0],
                [z_rad.sin(), z_rad.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ];

            // Combine the rotation matrices: R = Rz * Ry * Rx
            let rotation_xy = self.multiply_matrices_3x3(rotation_y, rotation_x);
            let rotation_xyz = self.multiply_matrices_3x3(rotation_z, rotation_xy);

            // Embed the 3x3 matrix into a 4x3 matrix
            [
                [rotation_xyz[0][0], rotation_xyz[0][1], rotation_xyz[0][2]],
                [rotation_xyz[1][0], rotation_xyz[1][1], rotation_xyz[1][2]],
                [rotation_xyz[2][0], rotation_xyz[2][1], rotation_xyz[2][2]],
                [0.0, 0.0, 0.0], // Fourth row is zero for a 4x3 matrix
            ]
        }

        /// Utility function: Multiply two 3x3 matrices
        fn multiply_matrices_3x3(&self, a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
            let mut result = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
                }
            }
            result
        }

        /// Helper function: Multiply a 4x3 matrix with a Vertex as translation vector.
        fn multiply_matrix_vector_4x3(&self, matrix: [[f64; 3]; 4], point: Vertex) -> Vertex {
            Vertex::new(
                matrix[0][0] * point.x
                    + matrix[0][1] * point.y
                    + matrix[0][2] * point.z
                    + matrix[3][0],
                matrix[1][0] * point.x
                    + matrix[1][1] * point.y
                    + matrix[1][2] * point.z
                    + matrix[3][1],
                matrix[2][0] * point.x
                    + matrix[2][1] * point.y
                    + matrix[2][2] * point.z
                    + matrix[3][2],
            )
        }

        /// Generate a panning transformation matrix
        pub fn get_pan_matrix_4x3(&self, right_amount: f64, up_amount: f64) -> [[f64; 3]; 4] {
            // Step 1: Compute the forward direction vector
            let forward = Vector3d::new(
                self.target.X - self.position.X,
                self.target.Y - self.position.Y,
                self.target.Z - self.position.Z,
            )
            .unitize_b();

            // Step 2: Compute the right direction vector (cross product of forward and up)
            let right = Vector3d::cross_product(&forward, &self.up).unitize_b();

            // Step 3: Compute the true up direction (orthogonalized)
            let up = Vector3d::cross_product(&right, &forward).unitize_b();

            // Step 4: Compute the translation
            let pan_translation =
                right.to_vertex() * (-right_amount) + (up.to_vertex() * up_amount);

            // Step 5: Construct the 4x3 pan matrix
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [pan_translation.x, pan_translation.y, pan_translation.z],
            ]
        }

        /// Apply panning to a set of 3D points
        pub fn pan_points_4x3(
            &self,
            points: &Vec<Vertex>,
            right_amount: f64,
            up_amount: f64,
        ) -> Vec<Vertex> {
            let pan_matrix = self.get_pan_matrix_4x3(right_amount, up_amount);
            points
                .iter()
                .map(|point| self.multiply_matrix_vector_4x3(pan_matrix, *point))
                .collect()
        }

        /// Apply panning to a set of mutable 3D points
        pub fn pan_points_mut_4x3(
            &self,
            points: &mut Vec<Vertex>,
            right_amount: f64,
            up_amount: f64,
        ) {
            let pan_matrix = self.get_pan_matrix_4x3(right_amount, up_amount);
            points.iter_mut().for_each(|point| {
                *point = self.multiply_matrix_vector_4x3(pan_matrix, *point);
            });
        }
        /// Combine multiple transformation matrices into one
        /// They have to be called from a stack vector using the macro `vec!` for that.
        pub fn combine_matrices_4x3(matrices: Vec<[[f64; 3]; 4]>) -> [[f64; 3]; 4] {
            // Start with the identity matrix
            let mut result = [
                [1.0, 0.0, 0.0], // First row
                [0.0, 1.0, 0.0], // Second row
                [0.0, 0.0, 1.0], // Third row
                [0.0, 0.0, 0.0], // Translation row
            ];

            // Multiply all matrices together in sequence
            for matrix in matrices {
                result = Self::multiply_matrices_4x3(result, matrix);
            }

            result
        }
        /// Helper function to multiply two 4x3 matrices
        pub fn multiply_matrices_4x3(a: [[f64; 3]; 4], b: [[f64; 3]; 4]) -> [[f64; 3]; 4] {
            let mut result = [[0.0; 3]; 4];

            // Multiply the rotation part (top 3x3 part of the matrices)
            for i in 0..3 {
                for j in 0..3 {
                    result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
                }
            }

            // Multiply the translation part
            for i in 0..3 {
                result[3][i] = a[3][0] * b[0][i] + a[3][1] * b[1][i] + a[3][2] * b[2][i] + b[3][i];
            }

            result
        }
    }
    pub mod coloring {
        /*
         * Color struct hold the RGB values in 3, 8 bit values and the total
         * absolute color value in a single u32 bit value
         * Most of the time expressed in Hexadecimal OxFFFFFFFF (4x8bit)
         * by other API.
         * - the absolute value and back ground value are optionals
         *   and can be cached for accelerated process.
         * - alpha is set to 1.0 (opaque) if not needed.
         * - every components are automatically cached and updated if
         *   they are claimed by runtime.
         * - everything is always kept Updated since structure is private
         *   and use only getter and setter as unique way to update
         *   the color description.
         * - alpha can be removed and RGB values are restored to the original
         *   opaque colors.
         * - draw back: it's a low level implementation and so alpha is always
         *   relative to a defined back ground color on function call.
         * */
        #[derive(Debug, Clone, Copy)]
        pub struct Color {
            red: u8,
            green: u8,
            blue: u8,
            alpha: f32,
            value: Option<u32>,
            bg_color: Option<u32>,
            original_value: Option<u32>,
        }
        impl Color {
            // Constructor A (initially without caching).
            pub fn new_rgb_fast(red: u8, green: u8, blue: u8) -> Self {
                Self {
                    red,
                    green,
                    blue,
                    alpha: 1.0,
                    value: None,
                    bg_color: None,
                    original_value: None,
                }
            }
            // Constructor B with cached absolute value.
            pub fn new_rgb(self, red: u8, green: u8, blue: u8) -> Self {
                let absolute_color = self.rgb_color(&red, &green, &blue);
                Self {
                    red,
                    green,
                    blue,
                    alpha: 1.0,
                    value: Some(absolute_color),
                    bg_color: None,
                    original_value: None,
                }
            }
            // Constructor C (with alpha).
            pub fn new_rgb_a(
                self,
                red: u8,
                green: u8,
                blue: u8,
                mut alpha: f32,
                background_color: u32,
            ) -> Self {
                if alpha < 1.0 {
                    let absolute_color =
                        self.rgba_color(&red, &green, &blue, &mut alpha, &background_color);
                    Self {
                        red: ((absolute_color >> 16) & 0xFF) as u8,
                        green: ((absolute_color >> 8) & 0xFF) as u8,
                        blue: (absolute_color & 0xFF) as u8,
                        alpha,
                        value: Some(absolute_color),
                        bg_color: Some(background_color),
                        original_value: Some(
                            (red as u32) << 16 | ((green as u32) << 8) | (blue as u32),
                        ),
                    }
                } else {
                    Self {
                        red,
                        green,
                        blue,
                        alpha,
                        value: Some((red as u32) << 16 | ((green as u32) << 8) | (blue as u32)),
                        bg_color: None,
                        original_value: None,
                    }
                }
            }

            /// Get absolute the absolute color value in u32.
            /// - compute & update alpha channel if nothing is cached..
            pub fn get_value(&mut self) -> u32 {
                if let Some(value) = self.value {
                    value // return value if cached.
                } else {
                    if self.alpha < 1.0 {
                        // Update and computed absolute color value from non opaque alpha
                        self.value = Some(self.rgba_color(
                            &self.red,
                            &self.green,
                            &self.blue,
                            &mut self.alpha,
                            &self.bg_color.unwrap(),
                        ));
                        // Backup original value.
                        self.original_value = Some(
                            (self.red as u32) << 16
                                | ((self.green as u32) << 8)
                                | (self.blue as u32),
                        );
                        // Update RGB description components from updated absolute value.
                        self.red = ((self.value.unwrap() >> 16) & 0xFF) as u8;
                        self.green = ((self.value.unwrap() >> 8) & 0xFF) as u8;
                        self.blue = ((self.value.unwrap()) & 0xFF) as u8;
                        self.value.unwrap() // return the computed absolute value.
                    } else {
                        // Update absolute value from RGB.
                        self.value = Some(
                            (self.red as u32) << 16
                                | ((self.green as u32) << 8)
                                | (self.blue as u32),
                        );
                        self.value.unwrap() // Return absolute 32bit value.
                    }
                }
            }

            /// Return Alpha Channel.
            pub fn get_alpha(self) -> f32 {
                self.alpha
            }

            /// Return Red component of value.
            pub fn get_red(self) -> u8 {
                self.red
            }

            /// Get Green.
            pub fn get_green(self) -> u8 {
                self.green
            }

            /// Get Blue.
            pub fn get_blue(self) -> u8 {
                self.blue
            }

            /// Mutate Color and Compute alpha from given background color.
            pub fn set_from_rgb_a_bg_components(
                &mut self,
                red: u8,
                green: u8,
                blue: u8,
                mut alpha: f32,
                bg_color: u32,
            ) {
                // Compute Alpha channel
                self.value = Some(self.rgba_color(&red, &green, &blue, &mut alpha, &bg_color));
                // Backup Original Value.
                self.original_value =
                    Some((self.red as u32) >> 16 | (self.green as u32) >> 8 | (self.green as u32));
                // Update RGBA_BG.
                self.red = red;
                self.green = green;
                self.blue = blue;
                self.alpha = alpha;
                self.bg_color = Some(bg_color);
            }

            /// Remove alpha channel.
            pub fn set_opaque(&mut self) {
                if self.alpha < 1.0 {
                    self.red = ((self.original_value.unwrap() >> 16) & 0xFF) as u8;
                    self.green = ((self.original_value.unwrap() >> 8) & 0xFF) as u8;
                    self.blue = ((self.original_value.unwrap()) & 0xFF) as u8;
                    self.alpha = 1.0;
                    self.original_value = None;
                    self.value = Some(
                        (self.red as u32) >> 16 | (self.green as u32) >> 8 | (self.green as u32),
                    );
                }
            }

            /// Mutate internals components and reset alpha state to opaque.
            pub fn set_from_absolute_value(&mut self, value: u32) {
                self.red = ((value >> 16) & 0xFF) as u8;
                self.green = ((value >> 8) & 0xFF) as u8;
                self.blue = (value & 0xFF) as u8;
                self.alpha = 1.0;
                self.original_value = None;
                self.value =
                    Some((self.red as u32) >> 16 | (self.green as u32) >> 8 | (self.green as u32));
            }

            /// Convert an rgb value to minifb 0rgb standard.
            fn rgb_color(self, red: &u8, green: &u8, blue: &u8) -> u32 {
                (*red as u32) << 16 | (*green as u32) << 8 | (*blue as u32)
            }

            /// Blend rgb value with alpha to back ground color.
            /// # Arguments
            /// - red u8
            /// - green u8
            /// - blue u8
            /// - alpha f32 from 0.0 (transparent) to 1.0 (opaque)
            /// - Background_color u32 (color to blend with)
            /// # Returns
            /// return an RGB color blended with the background color with the alpha from 0.0 to 1.0.
            fn rgba_color(
                self,
                red: &u8,
                green: &u8,
                blue: &u8,
                alpha: &mut f32,
                bg_color: &u32,
            ) -> u32 {
                // Ensure alpha is clamped between 0.0 and 1.0
                (*alpha) = (*alpha).clamp(0.0, 1.0);

                //TODO: a struct for Background color may be usefull
                // for caching in u8 form ....
                // Extract background RGB components as f32.
                let bg_r = ((bg_color >> 16) & 0xFF) as f32;
                let bg_g = ((bg_color >> 8) & 0xFF) as f32;
                let bg_b = (bg_color & 0xFF) as f32;

                // Blend each channel
                let blended_r = ((*alpha) * (*red) as f32 + (1.0 - (*alpha)) * bg_r).round() as u32;
                let blended_g =
                    ((*alpha) * (*green) as f32 + (1.0 - (*alpha)) * bg_g).round() as u32;
                let blended_b =
                    ((*alpha) * (*blue) as f32 + (1.0 - (*alpha)) * bg_b).round() as u32;
                (blended_r << 16) | (blended_g << 8) | blended_b
            }

            /// Public and static version of the private method.  
            /// Convert R,G,B to an absolute value u32.
            pub fn convert_rgb_color(red: u8, green: u8, blue: u8) -> u32 {
                (red as u32) << 16 | (green as u32) << 8 | (blue as u32)
            }

            /// Get absolute color value from RGB, alpha and Background Color chanels.
            pub fn convert_rgba_color(
                red: u8,
                green: u8,
                blue: u8,
                mut alpha: f32,
                bg_color: u32,
            ) -> u32 {
                // Ensure alpha is clamped between 0.0 and 1.0
                alpha = alpha.clamp(0.0, 1.0);

                //TODO: a struct for Background color may be usefull
                // for caching in u8 form ....
                // Extract background RGB components as f32.
                let bg_r = ((bg_color >> 16) & 0xFF) as f32;
                let bg_g = ((bg_color >> 8) & 0xFF) as f32;
                let bg_b = (bg_color & 0xFF) as f32;

                // Blend each channel
                let blended_r = (alpha * red as f32 + (1.0 - alpha) * bg_r).round() as u32;
                let blended_g = (alpha * green as f32 + (1.0 - alpha) * bg_g).round() as u32;
                let blended_b = (alpha * blue as f32 + (1.0 - alpha) * bg_b).round() as u32;
                (blended_r << 16) | (blended_g << 8) | blended_b
            }

            /// Provide a new RGB value blended in Background color from Alpha
            /// vale 0.0 (Transparent) to 1.0 Opaque.
            pub fn convert_rgb_with_background_and_alpha(
                foreground: u32, // 0xRRGGBB
                background: u32, // 0xRRGGBB
                alpha: f32,      // Transparency: 0.0 (transparent) to 1.0 (opaque)
            ) -> u32 {
                // Ensure alpha is clamped between 0.0 and 1.0
                let alpha = alpha.clamp(0.0, 1.0);

                // Extract RGB components of the foreground
                let fg_r = ((foreground >> 16) & 0xFF) as f32;
                let fg_g = ((foreground >> 8) & 0xFF) as f32;
                let fg_b = (foreground & 0xFF) as f32;

                // Extract RGB components of the background
                let bg_r = ((background >> 16) & 0xFF) as f32;
                let bg_g = ((background >> 8) & 0xFF) as f32;
                let bg_b = (background & 0xFF) as f32;

                // Blend each channel
                let blended_r = (alpha * fg_r + (1.0 - alpha) * bg_r).round() as u32;
                let blended_g = (alpha * fg_g + (1.0 - alpha) * bg_g).round() as u32;
                let blended_b = (alpha * fg_b + (1.0 - alpha) * bg_b).round() as u32;

                // Recombine into a single u32 color
                (blended_r << 16) | (blended_g << 8) | blended_b
            }
        }
    }
}
