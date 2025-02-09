/*
 *   Modules to organize the collection of object 3d by layers and visualize it
 *   from one camera or more with light and shaders.
 *    the main idea is to get:
 *    - a virtual space tracking geometry absolute location
 *    keeping them organised up to date by layer
 *    (the virtual space may be updated in real time)
 *    - a display pipe line space where one or more cameras can interpret
 *    the virtual space with light and shaders.
 */
use crate::cad_operations::draw::{self, draw_aa_line_with_thickness, draw_disc, draw_text};
use crate::cad_operations::geometry::{CPlane, Point3d, Vector3d};
use crate::cad_operations::intersection::clip_line;
use crate::cad_operations::transformation::{self};
use crate::cad_operations::utillity;
use crate::render_tools::rendering_object::{Mesh, MeshBox, Vertex};
use crate::render_tools::visualization_v3::coloring::Color;
use crate::render_tools::visualization_v4::Camera;
use core::f64;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::arch::global_asm;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;

// Updated Virtual_space.
// With true unique uid
////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub struct VirtualSpace {
    pub project_name: String,
    pub file_path: Option<String>,
    pub unit_scale: UnitScale,
    pub display: DisplayConfig,
    object_list: Vec<Arc<Mutex<Object3d>>>,
    pub layers: Vec<LayerVisibility>,
    scene_state: VirtualSpaceState,
    uid_list: Mutex<Vec<usize>>,
}

////////////////////////////////////////////////////////////////////////////////
impl VirtualSpace {
    /// Constructor of the main class.
    pub fn new(
        name: &str,
        file_path: Option<String>,
        unit_type: UnitScale,
        display: DisplayConfig,
    ) -> Self {
        Self {
            project_name: name.to_owned(),
            file_path,
            unit_scale: unit_type,
            display,
            object_list: Vec::new(),
            layers: Vec::new(),
            scene_state: VirtualSpaceState::SceneNeedUpdate,
            uid_list: Mutex::new(vec![0]),
        }
    }

    /// Add a new object to the list.
    pub fn add_obj(&mut self, mut object: Object3d) {
        // Increment the last unique id list number by 1 and add it to the list.
        if let Ok(mut m) = self.uid_list.lock() {
            let new_id = m[m.len() - 1] + 1usize;
            m.push(new_id);
            // Assign last unique id number to the object.
            object.id = new_id;
            // push into the vector.
            self.object_list.push(Arc::new(Mutex::new(object)));
            // acknowledge.
            self.scene_state = VirtualSpaceState::SceneNeedUpdate;
        }
    }

    /// Replace the `Displayable` in place for a specific object.
    pub fn replace_displayable_in_place_deprecated(
        &mut self,
        virtual_space_obj_index: usize,
        new_displayable_data: Displayable,
    ) {
        if let Some(object_arc) = self.object_list.get(virtual_space_obj_index) {
            let mut object = object_arc.lock().unwrap();

            // Push the current data to the undo stack if it exists
            if let Some(current_data) = object.data.clone() {
                object.undo_stack.push(current_data);
            }
            // Replace the current `data` with the new `Displayable`
            object.data = Some(Arc::new(Mutex::new(new_displayable_data)));
            object.update_date();
            self.scene_state = VirtualSpaceState::SceneNeedUpdate;
        }
    }

    /// Replace the `Displayable` in place for a specific object.
    pub fn replace_displayable_in_place(
        &mut self,
        virtual_space_obj_index: usize,
        new_displayable_data: Displayable,
    ) {
        // Attempt to get the object at the specified index.
        if let Some(object_arc) = self.object_list.get(virtual_space_obj_index) {
            // Try to lock the object. Handle potential poisoning gracefully.
            match object_arc.lock() {
                Ok(mut object) => {
                    // Push the current data to the undo stack if it exists.
                    if let Some(current_data) = object.data.take() {
                        object.undo_stack.push(current_data);
                    }
                    // Replace the current `data` with the new `Displayable`.
                    object.data = Some(Arc::new(Mutex::new(new_displayable_data)));

                    // Update the object's time stamp.
                    object.update_date();

                    // Mark the virtual space as needing an update.
                    self.scene_state = VirtualSpaceState::SceneNeedUpdate;
                }
                Err(e) => {
                    eprintln!(
                        "Failed to acquire lock on object at index {}: {:?}",
                        virtual_space_obj_index, e
                    );
                }
            }
        } else {
            eprintln!(
                "Object index {} out of bounds. Cannot replace Displayable.",
                virtual_space_obj_index
            );
        }
    }

    pub fn undo_displayable_in_place(&mut self, virtual_space_obj_index: usize) {
        // Check if the object exists at the specified index
        if let Some(object_arc) = self.object_list.get(virtual_space_obj_index) {
            match object_arc.lock() {
                Ok(mut object) => {
                    // Attempt to pop the undo stack
                    if let Some(undo_data) = object.undo_stack.pop() {
                        // Save the current `data` to the redo stack if it exists
                        if let Some(current_data) = object.data.take() {
                            object.redo_stack.push(current_data);
                        }
                        // Restore the previous state from the undo stack
                        object.data = Some(undo_data);
                        object.update_date();
                        self.scene_state = VirtualSpaceState::SceneNeedUpdate;
                    } else {
                        eprintln!("Undo failed: No more data in the undo stack.");
                    }
                }
                Err(_) => {
                    eprintln!(
                        "Error: Failed to lock the object at index {}. Mutex is poisoned.",
                        virtual_space_obj_index
                    );
                }
            }
        } else {
            eprintln!(
                "Error: No object found at index {}.",
                virtual_space_obj_index
            );
        }
    }

    pub fn undo_displayable_in_place_deprecated(&mut self, virtual_space_obj_index: usize) {
        let mut flg_action = false;
        if let Some(object_arc) = self.object_list.get(virtual_space_obj_index) {
            let mut object = object_arc.lock().unwrap();
            if let Some(undo_data) = object.undo_stack.pop() {
                if let Some(current_data) = object.data.clone() {
                    object.redo_stack.push(current_data);
                }
                // Restore the previous state
                object.data = Some(undo_data);
                object.update_date();
                self.scene_state = VirtualSpaceState::SceneNeedUpdate;
            } else {
                eprintln!("Nothing to undo... no change has occurred.");
            }
        }
    }

    pub fn redo_displayable_in_place(&mut self, virtual_space_obj_index: usize) {
        let mut flg_action = false;
        // Check if the object exists at the specified index
        if let Some(object_arc) = self.object_list.get(virtual_space_obj_index) {
            match object_arc.lock() {
                Ok(mut object) => {
                    // Attempt to pop the redo stack
                    if let Some(redo_data) = object.redo_stack.pop() {
                        // Save the current `data` to the undo stack if it exists
                        if let Some(current_data) = object.data.take() {
                            object.undo_stack.push(current_data);
                        }
                        // Restore the next state from the redo stack
                        object.data = Some(redo_data);
                        object.update_date();
                        self.scene_state = VirtualSpaceState::SceneNeedUpdate;
                    } else {
                        eprintln!("Redo failed: No more data in the redo stack.");
                    }
                }
                Err(_) => {
                    eprintln!(
                        "Error: Failed to lock the object at index {}. Mutex is poisoned.",
                        virtual_space_obj_index
                    );
                }
            }
        } else {
            eprintln!(
                "Error: No object found at index {}.",
                virtual_space_obj_index
            );
        }
    }

    pub fn redo_displayable_in_place_deprecated(&mut self, virtual_space_obj_index: usize) {
        if let Some(object_arc) = self.object_list.get(virtual_space_obj_index) {
            let mut object = object_arc.lock().unwrap();
            if let Some(redo_data) = object.redo_stack.pop() {
                if let Some(current_data) = object.data.clone() {
                    object.undo_stack.push(current_data);
                }
                // Restore the next state
                object.data = Some(redo_data);
                object.update_date();
                self.scene_state = VirtualSpaceState::SceneNeedUpdate;
            } else {
                eprintln!("Nothing to redo... no change has occurred.");
            }
        }
    }

    /// Clean the stack of empty elements.
    pub fn clean_stack(&mut self) {
        self.object_list.retain(|obj| {
            if let Some(m) = obj.lock().ok() {
                if m.data.is_none() {
                    // anti patern negate equality to remove the equality.
                    // Clean uid list.
                    if let Ok(mut m2) = self.uid_list.lock() {
                        m2.retain(|id| *id != m.id);
                    }
                }
                m.data.is_some()
            } else {
                true
            }
        });
        self.scene_state = VirtualSpaceState::SceneNeedUpdate;
    }
}

impl fmt::Display for VirtualSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let project_name = format!("{}", self.project_name);
        let path = if let Some(path) = &self.file_path {
            format!("{0}", path)
        } else {
            "None".to_string()
        };
        let unit_scale = self.unit_scale.to_string();
        let display_config = format!(
            "height: {}, width: {}, ratio: {}, raytrace enabled: ({})",
            self.display.display_resolution_height,
            self.display.display_resolution_width,
            self.display.display_ratio,
            self.display.raytrace
        );
        // Display object list
        let mut obj_list = String::new();
        obj_list.push_str(&format!(
            "Contains ({}) Object3d(s):\n",
            self.object_list.len()
        ));
        for (i, object_arc) in self.object_list.iter().enumerate() {
            let object = object_arc.lock().unwrap(); // Lock the mutex to access the object
            obj_list.push_str(&format!("  Index {}: {}\n", i, object));
        }
        // Display layers
        let mut layers_str = String::new();
        if self.layers.is_empty() {
            layers_str.push_str("No layers created yet.");
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                layers_str.push_str(&format!(
                    "  Layer {}: Visible: {}, Locked: {}, Color: {:?}\n",
                    i, layer.visibility, layer.lock, layer.color
                ));
            }
        }
        let state = self.scene_state.to_string();
        write!(
            f,
            "Virtual Space {{
    Project Name: '{}',
    File Path: {},
    Unit Scale: {},
    Display Config: {},
    Object3d List:
    {}
    Layers:
    {}
    State: {}
}}",
            project_name, path, unit_scale, display_config, obj_list, layers_str, state
        )
    }
}

// Updated Object3d
#[derive(Debug)]
pub struct Object3d {
    pub origin: CPlane,
    pub data: Option<Arc<Mutex<Displayable>>>,
    undo_stack: Vec<Arc<Mutex<Displayable>>>,
    redo_stack: Vec<Arc<Mutex<Displayable>>>,
    pub local_scale_ratio: f64,
    pub id: usize,
    pub last_change_date: String,
}

impl Object3d {
    /// Create an object ready to be stacked
    pub fn new(data: Displayable, origin: CPlane, local_scale_ratio: f64) -> Self {
        let data = Some(Arc::new(Mutex::new(data)));
        let id = 0;
        Self {
            id,
            data,
            origin,
            local_scale_ratio,
            last_change_date: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }
    /// Update last change date time.
    pub fn update_date(&mut self) {
        self.last_change_date = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    }
}

impl Clone for Object3d {
    fn clone(&self) -> Self {
        Self {
            origin: self.origin.clone(),
            data: self.data.clone(),
            undo_stack: self.undo_stack.clone(),
            redo_stack: self.redo_stack.clone(),
            local_scale_ratio: self.local_scale_ratio,
            id: self.id,
            last_change_date: self.last_change_date.clone(),
        }
    }
}

impl fmt::Display for Object3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format the `data` field
        let data_display = match &self.data {
            Some(opt) => match opt.lock() {
                Ok(dt) => dt.to_string(),
                Err(err) => {
                    format!("{err}")
                }
            },
            None => "None".to_string(),
        };
        // Format the undo and redo stacks
        let undo_stack_display = format!("({0} undo(s) elements.)", self.undo_stack.len());
        let redo_stack_display = format!("({0} redo(s) elements.)", self.redo_stack.len());
        // Display all the fields
        write!(
            f,
            " Object3d:{{
    origin: {0},
    data: {1},
    undo_stack: {2},
    redo_stack: {3},
    local_scale_ratio: {4},
    id: {5},
    last_change_date: {6},}}
",
            self.origin,
            data_display,
            undo_stack_display,
            redo_stack_display,
            self.local_scale_ratio,
            self.id,
            self.last_change_date,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub enum VirtualSpaceState {
    SceneNeedUpdate,
    SceneIsOk,
}
impl VirtualSpaceState {
    pub fn to_string(&self) -> String {
        match self {
            VirtualSpaceState::SceneNeedUpdate => {
                format!("SceneNeedUpdate")
            }
            VirtualSpaceState::SceneIsOk => format!("SceneIsOk"),
        }
    }
}

/// every thing that can be displayed for now
/// Curve are basically a set of points.
#[derive(Clone, Debug)]
pub enum Displayable {
    Point3d(Vec<Point3d>),
    Vector3d(Vec<Vector3d>),
    Vertex(Vec<Vertex>),
    Mesh(Mesh),
}
impl Displayable {
    pub fn to_string(&self) -> String {
        match &self {
            Displayable::Point3d(_) => format!("Object of type Point3d"),
            Displayable::Vector3d(_) => format!("Object of type Vector3d"),
            Displayable::Vertex(_) => format!("Object of type Vertex"),
            Displayable::Mesh(_) => format!("Object of type Mesh"),
        }
    }
}
// metric or imperial system reference.
#[derive(Clone, Debug)]
pub enum UnitScale {
    Millimeters,
    Centimeters,
    Meters,
    Inch,
}
impl UnitScale {
    pub fn to_string(&self) -> String {
        match self {
            UnitScale::Inch => {
                format!("Inch (imperial)")
            }
            UnitScale::Meters => {
                format!("Meter (metric)")
            }
            UnitScale::Millimeters => {
                format!("Millimeters (metric)")
            }
            UnitScale::Centimeters => {
                format!("Centimeters (metric)")
            }
        }
    }
}
// Config for the pipe_line Display thread.
#[derive(Debug)]
pub struct DisplayConfig {
    pub display_resolution_height: usize,
    pub display_resolution_width: usize,
    pub back_ground_color: u32,
    pub display_ratio: f64,
    pub camera_position: [[f64; 3]; 4], // special matrix format optimized
    // for visualization system.
    pub raytrace: bool,
}

impl DisplayConfig {
    pub fn new(height: usize, width: usize, back_ground_color: u32) -> Self {
        Self {
            display_resolution_height: height,
            display_resolution_width: width,
            display_ratio: (height as f64 / width as f64),
            back_ground_color: back_ground_color,
            // Create an identity matrix
            camera_position: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0], // last row are the swapped embedded translation vector.
            ],
            raytrace: false,
        }
    }
}

// A nested loop will send data to display pipeline from there.
#[derive(Debug)]
pub struct LayerVisibility {
    object3d_list_index: Vec<usize>,
    visibility: bool,
    color: Color,
    lock: bool,
}
/*
 * TODO:
 * notes:
 * for now start_display_pipeline use Object3d
 * from virtual space instance it should not
 * the function must render only object
 * from  data_to_render stack.
 * ( the idea is that the next thread process
 * an update data for rendering. )
 */

#[derive(Debug, Clone)]
pub struct DisplayPipeLine {
    pub data_to_render: Vec<Arc<Mutex<Object3d>>>,
    pub virtual_space: Arc<Mutex<VirtualSpace>>,
}

////////////////////////////////////////////////////////////////////////////////
impl DisplayPipeLine {
    /// Init the display pipe line.
    /// # Arguments
    /// take an Atomic Reference Counter Mutex of The Virtual_space.
    /// (for updating the scene state with hight priority).
    pub fn new(virtual_space_arc: Arc<Mutex<VirtualSpace>>) -> Self {
        Self {
            data_to_render: Vec::new(),
            virtual_space: virtual_space_arc,
        }
    }

    /// Feed display with rx reception data.
    /// # Arguments
    /// input data are internally preprocessed through pointers so no deep copy are involved.
    pub fn feed_data_pipe_entry(&mut self, rx_data: &Vec<Arc<Mutex<Object3d>>>) {
        // Format input in order to take only reference copy.
        let rx_data = DisplayPipeLine::format_rx_data(rx_data);
        // if there is no objects.
        if self.data_to_render.len() == 0 {
            self.data_to_render = rx_data;
        } else {
            // Update only the objects needed.
            for (i, d) in self.data_to_render.iter_mut().enumerate() {
                if let Some(mut a) = d.lock().ok() {
                    if let Some(b) = rx_data[i].lock().ok() {
                        if (a.last_change_date != b.last_change_date) && (a.id == b.id) {
                            *a = b.clone(); // Deep Copy of the data structure.
                        }
                    }
                }
            }
        }
        // Acknowledege the received data.
        self.send_data_received_state();
    }

    /// Format pointer stack for display Pipe Line Thread input
    /// by avoiding deep data copy but just passing pointers.
    /// # Returns
    /// a new Vec with Atomic References Counted pointer inside.
    pub fn format_rx_data(data_input: &Vec<Arc<Mutex<Object3d>>>) -> Vec<Arc<Mutex<Object3d>>> {
        data_input
            .iter()
            .map(|data| {
                data.clone() // copy only the pointer on the returned vector stack.
            })
            .collect()
    }

    /// Acknowledge that data_to_render is up to date.
    /// (intentionally private).
    fn send_data_received_state(&mut self) {
        match self.virtual_space.lock() {
            Ok(mut m) => {
                m.scene_state = VirtualSpaceState::SceneIsOk; // this action is only possible there.
            }
            Err(err) => {
                eprintln!("Cannot acknowledge scene update to \"SceneIsOk\" err:({err})");
            }
        }
    }

    pub fn start_display_pipeline(&mut self) {
        let thread_data = self.virtual_space.clone();
        if let Ok(m) = &thread_data.lock() {
            let windows_name = m.project_name.clone();
            let screen_height = m.display.display_resolution_height;
            let screen_width = m.display.display_resolution_width;
            let background_color = m.display.back_ground_color;
            // Init a widows 2D mini buffer class.
            let mut window = Window::new(
                &windows_name,
                screen_width,
                screen_height,
                WindowOptions::default(),
            )
            .unwrap_or_else(|e| {
                // panic on error (unwind stack and clean memory)
                panic!("{e}");
            });
            ////////////////////////////////////////////////////////////////////////////
            // A simple allocated array of u32 initialized at 0
            // representing the color and the 2d position of points.
            let mut buffer: Vec<u32> = vec![background_color; screen_width * screen_height];
            // Define the Display Unit initial Projection System.
            let mut camera = Camera::new(
                screen_width as f64,
                screen_height as f64,
                35.0,  // FOV (Zoom angle increase and you will get a smaller representation)
                0.5,   // Near clip plane
                100.0, // Far clip plane
            );
            window.set_target_fps(60); // limit to 60 fps max.
                                       // Scale transformation for the geometry.
            let scale_matrix = transformation::scaling_matrix_from_center_4x3(
                Vertex::new(0.0, 0.0, 0.0),
                0.5,
                0.5,
                0.5,
            );
            // Init angles of reference for the view experiment.
            let mut x_angle = 9.0;
            let mut zoom = 1.4;
            let mut z_angle = 67.5;
            let mut pan_x = 0.0;
            let mut pan_y = -0.15;
            // Camera matrix transformation.
            // initial position of the camera.
            let camera_position = camera.view_matrix;
            // Clear Console.
            println!("\x1b[2J");
            // Step angle in degrees.
            let step = 1.5;
            ////////////////////////////////////////////////////////////////////
            // Extract initial camera angles
            // (for visual initial camera vector direction as scene static reference).
            let cam_position = camera.position;
            let cam_target = camera.target;
            // Build Basic Components.
            let cam_dir = cam_target - cam_position;
            // Represent the camera vector.
            let reduce_factor = 0.1; // for visual display.
            let cam_eval_point = cam_target + -(cam_dir * reduce_factor);
            ////////////////////////////////////////////////////////////////////
            // Cache shared edge in memory for further usage.
            let mut shared_edge_map = HashMap::new();
            let mut mesh_vertices = Arc::new(Vec::new());
            let mut mesh_triangles = Arc::new(Vec::new());
            if let Ok(mesh) = m.object_list[0].lock() {
                if let Some(obj) = mesh.data.clone() {
                    if let Ok(mut m) = obj.lock() {
                        if let Displayable::Mesh(ref mut mesh) = *m {
                            // Update the size of the imported mesh (scale down by 50%).
                            mesh.vertices =
                                transformation::transform_points_4x3(&scale_matrix, &mesh.vertices);
                            // Cache the mapping of shared edge by adjacent triangles.
                            shared_edge_map = mesh.find_shared_edges();
                            // Extract vertices and triangle as smart pointers (avoiding deep copy later in the code)
                            mesh_vertices = Arc::new(mesh.vertices.clone());
                            mesh_triangles = Arc::new(mesh.triangles.clone());
                        }
                    }
                }
            }
            ////////////////////////////////////////////////////////////////////
            // Update Camera flg.
            let mut update_flg = true; // update the first frame.
            let mut silhouette_flg = true;
            let mut grid_flg = true;
            let mut cam_axis_flg = false;
            let mut test_flg = false;
            let mut mesh_flg = true;
            println!("\x1b[1;0H\x1b[2K\r-> Press arrows to rotate the geometry, (z + Up) or (z + Down) for zooming or (Space + Direction) to pan the camera. ('g','c','s','m' or 't') for toggling display option on/off");
            ////////////////////////////////////////////////////////////////////
            while window.is_open() && !window.is_key_down(Key::Escape) {
                if window.is_key_down(Key::Space)
                    && window.is_key_pressed(Key::Up, minifb::KeyRepeat::No)
                {
                    pan_y -= 0.05;
                    update_flg = true;
                } else if window.is_key_down(Key::Space)
                    && window.is_key_pressed(Key::Down, minifb::KeyRepeat::No)
                {
                    pan_y += 0.05;
                    update_flg = true;
                } else if window.is_key_down(Key::Space)
                    && window.is_key_pressed(Key::Left, minifb::KeyRepeat::No)
                {
                    pan_x += 0.05;
                    update_flg = true;
                } else if window.is_key_down(Key::Space)
                    && window.is_key_pressed(Key::Right, minifb::KeyRepeat::No)
                {
                    pan_x -= 0.05;
                    update_flg = true;
                } else if window.is_key_down(Key::Z)
                    && window.is_key_pressed(Key::Up, minifb::KeyRepeat::Yes)
                {
                    zoom += 0.05;
                    update_flg = true;
                } else if window.is_key_down(Key::Z)
                    && window.is_key_pressed(Key::Down, minifb::KeyRepeat::Yes)
                {
                    zoom -= 0.05;
                    update_flg = true;
                } else if window.is_key_pressed(Key::Left, minifb::KeyRepeat::Yes) {
                    println!("\x1b[2;0H\x1b[2K\rkey Left pressed");
                    z_angle += step;
                    update_flg = true;
                } else if window.is_key_pressed(Key::Right, minifb::KeyRepeat::Yes) {
                    println!("\x1b[2;0H\x1b[2K\rkey Right pressed");
                    z_angle -= step;
                    update_flg = true;
                } else if window.is_key_pressed(Key::Up, minifb::KeyRepeat::Yes) {
                    println!("\x1b[2;0H\x1b[2K\rkey Up pressed");
                    x_angle += step;
                    update_flg = true;
                } else if window.is_key_pressed(Key::Down, minifb::KeyRepeat::Yes) {
                    println!("\x1b[2;0H\x1b[2K\rkey Down pressed");
                    x_angle -= step;
                    update_flg = true;
                } else if window.is_key_pressed(Key::S, KeyRepeat::No) {
                    silhouette_flg = !silhouette_flg;
                    update_flg = true;
                } else if window.is_key_pressed(Key::G, KeyRepeat::No) {
                    grid_flg = !grid_flg;
                    update_flg = true;
                } else if window.is_key_pressed(Key::T, KeyRepeat::No) {
                    test_flg = !test_flg;
                    update_flg = true;
                } else if window.is_key_pressed(Key::C, KeyRepeat::No) {
                    cam_axis_flg = !cam_axis_flg;
                    update_flg = true;
                } else if window.is_key_pressed(Key::M, KeyRepeat::No) {
                    mesh_flg = !mesh_flg;
                    update_flg = true;
                }
                ////////////////////////////////////////////////////////////////
                // Update frame buffer only if something as changed or (at first frame);
                if update_flg {
                    camera.orbit_around_world_z(
                        &camera_position,
                        x_angle,
                        z_angle,
                        zoom,
                        pan_x,
                        pan_y,
                    );
                    ////////////////////////////////////////////////////////////
                    // Format background color.
                    for pixel in buffer.iter_mut() {
                        *pixel = background_color; // Stet the bg color.
                    }
                    // Display camera matrix on Console with rounded digits.
                    println!(
                    "\x1b[3;0H\x1b[2K\r[{0:>6.3},{1:>6.3},{2:>6.3},{3:>6.3}]\x1b[4;0H\x1b[2K\r[{4:>6.3},{5:>6.3},{6:>6.3},{7:>6.3}]\
                    \x1b[5;0H\x1b[2K\r[{8:>6.3},{9:>6.3},{10:>6.3},{11:>6.3}]\x1b[6;0H\x1b[2K\r[{12:>6.3},{13:>6.3},{14:>6.3},{15:>6.3}]",
                    camera.view_matrix[0][0],camera.view_matrix[0][1],camera.view_matrix[0][2],camera.view_matrix[0][3],
                    camera.view_matrix[1][0],camera.view_matrix[1][1],camera.view_matrix[1][2],camera.view_matrix[1][3],
                    camera.view_matrix[2][0],camera.view_matrix[2][1],camera.view_matrix[2][2],camera.view_matrix[2][3],
                    camera.view_matrix[3][0],camera.view_matrix[3][1],camera.view_matrix[3][2],camera.view_matrix[3][3],
                );
                    if grid_flg {
                        ////////////////////////////////////////////////////////////////
                        // Draw an Unit grid test.
                        let o = Point3d::new(0.0, 0.0, 0.0);
                        let x = o + Point3d::new(0.1, 0.0, 0.0);
                        let y = o + Point3d::new(0.0, 0.1, 0.0);
                        let p = CPlane::new_origin_x_aligned_y_oriented(&o, &x, &y);
                        let mut global_alpha = 1.0;
                        let mut grid_length = 0.8;
                        let cam_distance = camera.position.magnitude();
                        if cam_distance < (grid_length / 2.0) + 0.2 {
                            grid_length = 0.2;
                            if cam_distance < 0.3 {
                                // ilerp function is feed from fixed values (in that case) so 
                                // better there to take advantage of the zero cost abstraction of the .unwrap() method.
                                global_alpha =  utillity::ilerp(0.15, 0.3, cam_distance).unwrap();
                            }
                        }
                        if global_alpha > 0.0 {
                            draw::draw_unit_grid_system(
                                &mut buffer,
                                screen_width,
                                screen_height,
                                background_color,
                                global_alpha,
                                &p,
                                &camera,
                                None,
                                grid_length,
                                grid_length,
                                0.1,
                            );
                        }
                        // Test for a gimball graphic position.
                        let o = Point3d::new(0.3, -0.2, 0.0);
                        let x = o + Point3d::new(0.1, 0.0, 0.0);
                        let y = o + Point3d::new(0.0, 0.1, 0.0);
                        let p2 = CPlane::new_origin_x_aligned_y_oriented(&o, &x, &y);
                        draw::draw_gimball_from_plane(
                            &mut buffer,
                            screen_width,
                            screen_height,
                            background_color,
                            &p2,
                            &camera,
                            None,
                            0.15,
                            1.0,
                            true,
                        );
                    }
                    ////////////////////////////////////////////////////////////////
                    // Display vertex of imported mesh. (wire frame not available yet) a GPU
                    // acceleration would be beneficial for that.
                    ////////////////////////////////////////////////////////////////
                    // Get points.
                    if let Ok(mesh) = m.object_list[0].lock() {
                        if let Some(obj) = mesh.data.clone() {
                            if let Ok(mut m) = obj.lock() {
                                if let Displayable::Mesh(ref mut mesh) = *m {
                                    if mesh_flg {
                                        // Project mesh vertices on screen space coordinates.
                                        let projected_points =
                                            camera.project_a_list_of_points(&mesh.vertices);
                                        projected_points.iter().for_each(|point| {
                                            buffer[point.1 * screen_width + point.0] = 0x00004e;
                                        });
                                    }
                                    if silhouette_flg {
                                        let buffer_ptr = Arc::new(Mutex::new(&mut buffer));
                                        // Extract mesh silhouette dynamically from camera orientation.
                                        // ( The use of a cached shared edge map help to reduce
                                        // computation time )
                                        let border_edges = mesh.extract_silhouette_simd_beta(
                                            &mesh_vertices,
                                            &mesh_triangles,
                                            &shared_edge_map,
                                            &camera.get_camera_direction(),
                                        );
                                        // Project each edge points in parallel on screen space.
                                        let projected_edges_points: Vec<((f64, f64), (f64, f64))> =
                                            border_edges
                                                .par_iter()
                                                .map(|edge| {
                                                    (
                                                        camera.project_maybe_outside(&edge.0),
                                                        camera.project_maybe_outside(&edge.1),
                                                    )
                                                })
                                                .collect();
                                        // Draw Edge in parallel (each lines by thread.).
                                        projected_edges_points.par_iter().for_each({
                                            let buffer_ptr_instance = buffer_ptr.clone();
                                            move |&edge| {
                                                if let Some(pt) = clip_line(
                                                    edge.0,
                                                    edge.1,
                                                    screen_width,
                                                    screen_height,
                                                ) {
                                                    if let Some(mut mutex) =
                                                        buffer_ptr_instance.lock().ok()
                                                    {
                                                        draw::draw_line(
                                                            &mut (*mutex),
                                                            screen_width,
                                                            (pt.0 .0 as usize, pt.0 .1 as usize),
                                                            (pt.1 .0 as usize, pt.1 .1 as usize),
                                                            0xFFD700,
                                                        );
                                                    }
                                                }
                                            }
                                        });
                                    }
                                }
                            }
                        }
                    }
                    if test_flg {
                        ////////////////////////////////////////////////////////////////
                        // Draw a parametric mesh Box.
                        // (graphic wire frame are just temporary for mesh design)
                        // a method for mesh will be implemented soon.
                        ////////////////////////////////////////////////////////////////
                        let origin = Vertex::new(-0.05, -0.2, 0.025);
                        let mut dir_u = Vertex::new(1.0, 0.0, 0.0);
                        let mut dir_v = Vertex::new(0.0, 1.0, 0.0);
                        let m_box = MeshBox::new(
                            &mut buffer,
                            screen_width,
                            screen_height,
                            &camera,
                            None,
                            &origin,
                            &mut dir_u,
                            &mut dir_v,
                            0.1,
                            0.1,
                            0.1,
                            2,
                            2,
                            2,
                        )
                        .to_mesh();
                        /////////////////////////////////////////////////////////////////////////////////////
                        // Display Mesh triangles Normals   /////////////////////////////////////////////////
                        let box_normals_list = m_box.extract_faces_normals_vectors();
                        for (i, norm) in box_normals_list.iter().enumerate() {
                            let p1 = camera.project_maybe_outside(&norm.0);
                            let p2 = norm.0 + (norm.1 * 0.02); // Reduce the normals distances display.
                            let p2 = camera.project_maybe_outside(&p2);
                            if let Some(pt) = clip_line(p1, p2, screen_width, screen_height) {
                                draw_aa_line_with_thickness(
                                    &mut buffer,
                                    screen_width,
                                    pt.0,
                                    pt.1,
                                    3,
                                    0x00FF00,
                                );
                                draw_text(
                                    &mut buffer,
                                    screen_height,
                                    screen_width,
                                    p1.0 as usize,
                                    p1.1 as usize,
                                    format!("{i}").as_str(),
                                    2,
                                    0x0,
                                );
                            }
                        }
                        ////////////////////////////////////////////////////////////////
                        // Test for an arbitrary input vector (for silhouette extraction).
                        ////////////////////////////////////////////////////////////////
                        let pt1 = Vertex::new(-0.1, -0.1, -0.1); // Position
                        let pt2 = Vertex::new(0.0, 0.0, 0.0); // Target
                        println!(
                        "\x1b[2K\rTest vector (for silhouette extraction):{0:?}, Test target:{1:?}",
                        pt1, pt2
                    );
                        let test_dir = (pt2 - pt1).normalize();
                        let pt2 = pt1 + (test_dir * 0.1); // clamp at 10% of the target distance.
                        let pt1 = camera.project_maybe_outside(&pt1);
                        let pt2 = camera.project_maybe_outside(&pt2);
                        if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                            draw_aa_line_with_thickness(
                                &mut buffer,
                                screen_width,
                                pt.0,
                                pt.1,
                                3,
                                0x0000FF,
                            );
                        }
                        // This sim to work with a regular vector. (now i just need to compute the camera position.)
                        let border_edges =
                            m_box.extract_silhouette_vb(&m_box.find_shared_edges(), &test_dir);
                        for (i, edge) in border_edges.iter().enumerate() {
                            let pt1 = camera.project_maybe_outside(&edge.0);
                            let pt2 = camera.project_maybe_outside(&edge.1);
                            if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                                draw_aa_line_with_thickness(
                                    &mut buffer,
                                    screen_width,
                                    pt.0,
                                    pt.1,
                                    3,
                                    0x0000FF,
                                );
                            }
                            //println!("\x1b[{0};0H\x1b[2K\r{1} iteration:{2}", i + 6, edge.2, i);
                        }
                        ////////////////////////////////////////////////////////////////
                        // Test triangle indices localization.
                        ////////////////////////////////////////////////////////////////
                        let triangle_id_to_test = 10;
                        if let Some(pt) = camera.project_without_depth(
                            &m_box.triangles[triangle_id_to_test].center_to_vertex(&m_box.vertices),
                        ) {
                            draw_disc(
                                &mut buffer,
                                screen_width,
                                screen_height,
                                pt.0,
                                pt.1,
                                5,
                                0x0000FF,
                                1,
                            );
                        }
                        /*
                                        // Solve culling faces orientation.
                                        println!(
                                            "\x1b[2K\r{0:?} dot{1}",
                                            m_box.triangles[1],
                                            m_box.triangles[1].normal.dot(&cam_dir)
                                        );
                                        println!(
                                            "\x1b[2K\r{0:?} dot{1}",
                                            m_box.triangles[10],
                                            m_box.triangles[10].normal.dot(&cam_dir)
                                        );

                                        println!(
                                            "\x1b[2K\rCam direction: {0:?} cam position: {1:?}",
                                            cam_dir,
                                            cam_position,
                                        );

                        */
                    }
                    if cam_axis_flg {
                        ////////////////////////////////////////////////////////////////
                        //          Initial camera direction representation.  (in Red)
                        ////////////////////////////////////////////////////////////////
                        let pt1 = camera.project_maybe_outside(&cam_eval_point);
                        let pt2 = camera.project_maybe_outside(&cam_target);
                        if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                            draw_aa_line_with_thickness(
                                &mut buffer,
                                screen_width,
                                pt.0,
                                pt.1,
                                3,
                                0xFF0000,
                            );
                        }
                        ////////////////////////////////////////////////////////////////
                        // Draw Camera Base components.       Testing. (no good result yet.)
                        let pt1 =
                            camera.project_maybe_outside(&(camera.cam_right.to_vertex() * 0.2));
                        let pt2 = camera.project_maybe_outside(&cam_target);
                        if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                            draw_aa_line_with_thickness(
                                &mut buffer,
                                screen_width,
                                pt.0,
                                pt.1,
                                3,
                                0x0,
                            );
                        }
                        // The camera position with partial panning.
                        println!(
                    "\x1b[2K\rKey Board input(s) -> Angle(x:{0:^6.1},y:{1:^6.1}),Zoom:({2:^3.1}),Camera Panning:(x:{3:^3.2},y:{4:^4.2}))",
                    x_angle, z_angle, zoom,pan_x,pan_y
                );
                        println!(
                            "\x1b[2K\r- Camera position:{0}, Target:{1}",
                            camera.position, camera.target
                        );
                        println!(
                            "\x1b[2K\r- Camera orientation: Up:{0}, Right:{1}, Forward:{2}",
                            camera.cam_up, camera.cam_right, camera.cam_forward
                        );
                        // Plot the camera base vectors from camera parameters.
                        // Cam right (red).
                        let pt1 = camera.project_maybe_outside(&camera.target);
                        let pt2 = camera.project_maybe_outside(
                            &(camera.target + (camera.cam_right.to_vertex() * 0.1)),
                        );
                        if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                            draw_aa_line_with_thickness(
                                &mut buffer,
                                screen_width,
                                pt.0,
                                pt.1,
                                3,
                                0xFF0000,
                            );
                        }
                        // Cam up (yellow).
                        let pt1 = camera.project_maybe_outside(&camera.target);
                        let pt2 = camera.project_maybe_outside(
                            &(camera.target + (camera.cam_up.to_vertex() * 0.1)),
                        );
                        if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                            draw_aa_line_with_thickness(
                                &mut buffer,
                                screen_width,
                                pt.0,
                                pt.1,
                                3,
                                0xFFD700,
                            );
                        }
                        // Cam forward (blue almost invisible since the line is a single dot projection).
                        let pt1 = camera.project_maybe_outside(&camera.target);
                        let pt2 = camera.project_maybe_outside(
                            &(camera.target + (camera.cam_forward.to_vertex() * 0.1)),
                        );
                        if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                            draw_aa_line_with_thickness(
                                &mut buffer,
                                screen_width,
                                pt.0,
                                pt.1,
                                3,
                                0x0000FF,
                            );
                        }
                        println!(
                            "\x1b[2K\r-> Camera target from view matrix projection {}",
                            camera.get_camera_target_zoom_insensitive()
                        );
                    }
                }
                // Reset flag for next loop.
                update_flg = false;
                window
                    .update_with_buffer(&buffer, screen_width, screen_height)
                    .unwrap();
            }
        };
    }
}

////////////////////////////////////////////////////////////////////////////////
/*
 * this is a template prototype of the scripted runtime
 * from json Deserialization.
 */
////////////////////////////////////////////////////////////////////////////////
pub mod runtime_concept {
    pub fn runtime() {
        // Runtime Stack.
        let mut runtime: Vec<FunctionPointer> = Vec::new();

        // Push from Json Deserialization. (function + parameters into a struct)
        runtime.push(FunctionPointer::Magnetude(magnetude));
        runtime.push(FunctionPointer::Inversesqrt(inverse_sqrt));
        runtime.push(FunctionPointer::SomeOperation(some_operation));
        runtime.push(FunctionPointer::Displaystruct(display_struct));

        // Feed parameters from struct related to the stack order.
        for (index, fpointer) in runtime.iter().enumerate() {
            match fpointer {
                FunctionPointer::Magnetude(f) => {
                    println!("--->id:{index} fcall:{:?}", f(3.0, 4.0, 0.0));
                }
                FunctionPointer::Inversesqrt(f) => {
                    println!("--->id:{index} fcall:{:?}", f(81.0));
                }
                FunctionPointer::SomeOperation(f) => {
                    println!("--->id:{index} fcall:{:?}", f(2, 2));
                }
                FunctionPointer::Displaystruct(f) => {
                    let v = Vector2d { x: 4, y: 7 };
                    println!("--->id:{index} fcall:{:?}", f(&v));
                }
            }
        }
    }
    // my lib....
    pub fn magnetude(x: f64, y: f64, z: f64) -> f64 {
        (x * x + y * y + z * z).sqrt()
    }
    pub fn inverse_sqrt(value: f64) -> f64 {
        1.0 / (value.sqrt())
    }
    fn some_operation<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
        a + b
    }
    pub fn display_struct(input: &Vector2d) -> String {
        format!("{input:?}")
    }
    #[derive(Debug)]
    pub struct Vector2d {
        x: u8,
        y: u8,
    }
    // Using ia to process my lib in a enum wrapper.
    // or wrap relevant function by hand through procedural
    // wrapper function.
    #[derive(Debug)]
    enum FunctionPointer {
        Magnetude(fn(x: f64, y: f64, z: f64) -> f64),
        Inversesqrt(fn(value: f64) -> f64),
        SomeOperation(fn(a: u8, b: u8) -> u8), // Generic have to by call contextually.
        Displaystruct(fn(&Vector2d) -> String),
    }
}
// serialize/deserialize json process.
pub mod json {
    use serde::{Deserialize, Serialize};
    use std::fs;
    #[derive(Serialize, Deserialize, Debug)]
    pub struct Person {
        name: String,
        age: u8,
        email: String,
    }
    pub fn run_json() {
        let person = Person {
            name: "John Doe".to_string(),
            age: 30,
            email: "john.doe@example.com".to_string(),
        };

        // Serialize the struct to a JSON string
        let json_string = serde_json::to_string(&person).unwrap();
        println!("Serialized JSON: {}", json_string);

        // Write file
        fs::write("./user_data/person.json", json_string).expect("Unable to write file");

        // Read file
        let json_data = fs::read_to_string("person.json").expect("Unable to read file");
        let json_data = r#"{"name":"John Doe","age":30,"email":"john.doe@example.com"}"#;

        // Deserialize the JSON string into a struct with error handling.
        let result: Result<String, serde_json::Error> = serde_json::from_str(json_data);
        match result {
            Ok(person) => println!("Deserialized struct: {:?}", person),
            Err(e) => eprintln!("Error deserializing JSON: {}", e),
        }
        println!("Deserialized struct: {:?}", person);
    }
}
