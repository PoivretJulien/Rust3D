// Virtual space will have the responsibility:
// = to have a data structure to hold user static constants
//   like:
//          - display resolution.
//          - application setting related to performance/memory
// - other file config for persistent parameters like:
//          - name of the project.
//          - file path.
//          - camera position.
//          - unit scale
//          - undo/redo stack:
//                 - this may be object constraint to a displayable(trait) for
//                   position/scale information before modification from the
//                   moment where the file is loaded,
//                   for that each objects should be bound to a stack index id
//                   where displayable(trait) is stored on...
//                   that will be named the "virtual_space whole list object".
//                 - the undo/redo stack by object
//                   will just keep track of that list by keeping 3 or 4 displayable(trait)
//                   state and swap back the object on that list on user call.
//           - so, the virtual_space object whole list will store each objects on a unique
//             index id where undo redo stack will keep track of history state.
//           - a possible layer interpretation of the virtual_space
//             where a struct list will keep track of the layer visibility
//             name id and al displayable(trait) visual characteristic
//             (shader, color, wireframe_color, opacity) to feed into
//             the pipe_line display runtime parameters.
//           - a function pointer list of selected procedural function for user
//             interaction and runtime script via json file.
// - loading a geometry file into the whole list object and put it in the center of the virtual_space
// - each objects of the whole list object may be stored in a struct where position
// - scale transformation and undo redo where stack is stored.
// - once the virtual space trigger an enum state ready
//   the display pipe line will just start to interpret the virtual_space
//   (from another thread) the whole virtual_space object list
//   (by matrix operations from user input and rasterize / raytrace
//    the virtual_space)
//  - Before sending to display pipe line iter from Layer Visibility object and
//    apply related parameters in function.

use crate::render_tools::rendering_object::{Mesh, MeshBox, MeshPlane, Vertex};
use crate::render_tools::visualization_v3::coloring::*;
use crate::render_tools::visualization_v4::Camera;
use crate::rust3d::draw::{
    self, draw_aa_line, draw_aa_line_with_thickness, draw_disc, draw_gimball_from_plane, draw_text,
};
use crate::rust3d::intersection::clip_line;
use crate::rust3d::transformation;
use crate::rust3d::{self, geometry::*};
use core::f64;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;

// Updated Virtual_space.
// With true unique uuid
////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub struct Virtual_space {
    pub project_name: String,
    pub file_path: Option<String>,
    pub unit_scale: Unit_scale,
    pub display: Display_config,
    object_list: Vec<Arc<Mutex<Object3d>>>,
    pub layers: Vec<LayerVisibility>,
    scene_state: VirtualSpaceState,
    uid_list: Mutex<Vec<usize>>,
}

////////////////////////////////////////////////////////////////////////////////
impl Virtual_space {
    /// Constructor of the main class.
    pub fn new(
        name: &str,
        file_path: Option<String>,
        unit_type: Unit_scale,
        display: Display_config,
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
            // acknowledege.
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

                    // Update the object's timestamp.
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

impl fmt::Display for Virtual_space {
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
pub enum Unit_scale {
    Millimeters,
    Centimeters,
    Meters,
    Inch,
}
impl Unit_scale {
    pub fn to_string(&self) -> String {
        match self {
            Unit_scale::Inch => {
                format!("Inch (imperial)")
            }
            Unit_scale::Meters => {
                format!("Meter (metric)")
            }
            Unit_scale::Millimeters => {
                format!("Millimeters (metric)")
            }
            Unit_scale::Centimeters => {
                format!("Centimeters (metric)")
            }
        }
    }
}
// Config for the pipe_line Display thread.
#[derive(Debug)]
pub struct Display_config {
    pub display_resolution_height: usize,
    pub display_resolution_width: usize,
    pub back_ground_color: u32,
    pub display_ratio: f64,
    pub camera_position: [[f64; 3]; 4], // special matrix format optimized
    // for visualization system.
    pub raytrace: bool,
}

impl Display_config {
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
struct LayerVisibility {
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
    pub virtual_space: Arc<Mutex<Virtual_space>>,
}

////////////////////////////////////////////////////////////////////////////////
impl DisplayPipeLine {
    /// Init the display pipe line.
    /// # Arguments
    /// take an Atomic Reference Counter Mutex of The Virtual_space.
    /// (for updating the scene state with hight priority).
    pub fn new(virtual_space_arc: Arc<Mutex<Virtual_space>>) -> Self {
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

    /// Acknoledge that data_to_render is up to date.
    /// (intentionally private).
    fn send_data_received_state(&mut self) {
        match self.virtual_space.lock() {
            Ok(mut m) => {
                m.scene_state = VirtualSpaceState::SceneIsOk; // this action is only possible there.
            }
            Err(err) => {
                eprintln!("Cannot acknowledege scene update to \"SceneIsOk\" err:({err})");
            }
        }
    }

    /*
       Start the conduit init the fb resolution, the buffer memory space,
       load/update the geometry if virtual state is "SceneNeedUpdate",then preprocess
       transformation of the geometry from the user input KEY and finaly
       if conduit need update from the previous loop, project points
       mutate and update the buffer to the screen.
    */
    pub fn start_display_pipeline(&mut self) {
        // minifb must be lunch in the main thread (i was not aware about that) so the overall
        // runtime will be simply flipped:
        // - a thread will be allocated for virtualspace
        // (where geometric interactions will be computed.)
        // - the main thread will take care of the display pipe line with mini fb.
        // i'm focusing on graphical tools for now and try to learn some techniques.
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
                Point3d::new(0.0, -1.0, -0.3), // Camera position (1 is the max value)
                Point3d::new(0.0, 0.0, 0.0), // Camera target ( relative to position must be 0,0,0 )
                Vector3d::new(0.0, 0.0, 1.0), // Camera up vector (for inner cross product operation usually Y=1)
                screen_width as f64,
                screen_height as f64,
                35.0,  // FOV (Zoom angle increase and you will get a smaller representation)
                0.5,   // Near clip plane
                100.0, // Far clip plane
            );
            window.set_target_fps(60); // limit to 25 fps max.
                                       // Scale transformation for the geometry.
            let scale_matrix = transformation::scaling_matrix_from_center_4x3(
                Vertex::new(0.0, 0.0, 0.0),
                0.5,
                0.5,
                0.5,
            );
            let mut z_angle = 0.0;
            let mut x_angle = 0.0;
            // Camera matrix transformation.
            let camera_position = camera.view_matrix;
            let mut orbit_matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let pan_matrix = camera.transform_camera_matrix_pan(0.0, -0.1);
            println!("\x1b[2J");
            println!("\x1b[1;0H\x1b[2K\r-> Press arrows of the keys board to rotate the geometry.");
            let step = 1.0;
            while window.is_open() && !window.is_key_down(Key::Escape) {
                for pixel in buffer.iter_mut() {
                    *pixel = background_color; // Stet the bg color.
                }
                // Catch user input keys.
                if window.is_key_pressed(Key::Left, minifb::KeyRepeat::Yes) {
                    println!("\x1b[1;0H\x1b[2K\rkey Left pressed");
                    z_angle += step;
                    orbit_matrix =
                        transformation::rotation_matrix_from_angles(x_angle, 0.0, z_angle);
                }
                if window.is_key_pressed(Key::Right, minifb::KeyRepeat::Yes) {
                    println!("\x1b[1;0H\x1b[2K\rkey Right pressed");
                    z_angle -= step;
                    orbit_matrix =
                        transformation::rotation_matrix_from_angles(x_angle, 0.0, z_angle);
                }
                if window.is_key_pressed(Key::Up, minifb::KeyRepeat::Yes) {
                    println!("\x1b[1;0H\x1b[2K\rkey Up pressed");
                    x_angle += step;
                    orbit_matrix =
                        transformation::rotation_matrix_from_angles(x_angle, 0.0, z_angle);
                }
                if window.is_key_pressed(Key::Down, minifb::KeyRepeat::Yes) {
                    println!("\x1b[1;0H\x1b[2K\rkey Down pressed");
                    x_angle -= step;
                    orbit_matrix =
                        transformation::rotation_matrix_from_angles(x_angle, 0.0, z_angle);
                }
                ////////////////////////////////////////////////////////////////
                // Update Camera Position.
                camera.view_matrix = transformation::combine_matrices(vec![
                    camera_position,
                    pan_matrix,
                    orbit_matrix,
                ]);

                //camera.apply_transformations(vec![camera_position,pan_matrix,orbit_matrix]);
                // Display camera matrix on std out console.
                println!(
                    "\x1b[2;0H\x1b[2K\r{0:?}\x1b[3;0H\x1b[2K\r{1:?}\x1b[4;0H\x1b[2K\r{2:?}\x1b[5;0H\x1b[2K\r{3:?}",
                    camera.view_matrix[0], camera.view_matrix[1], camera.view_matrix[2],camera.view_matrix[3]
                );
                ////////////////////////////////////////////////////////////////
                // Draw an Unit grid test.
                let o = Point3d::new(0.0, 0.0, 0.0);
                let x = o + Point3d::new(0.1, 0.0, 0.0);
                let y = o + Point3d::new(0.0, 0.1, 0.0);
                let p = CPlane::new_origin_x_aligned_y_oriented(&o, &x, &y);
                draw::draw_unit_grid_system(
                    &mut buffer,
                    screen_width,
                    screen_height,
                    background_color,
                    &p,
                    &camera,
                    None,
                    1.0,
                    1.0,
                    0.1,
                );
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
                ////////////////////////////////////////////////////////////////
                if true {
                    // Get points.
                    if let Ok(mesh) = m.object_list[0].lock() {
                        if let Some(obj) = mesh.data.clone() {
                            if let Ok(mut m) = obj.lock() {
                                if let Displayable::Mesh(ref mut mesh) = *m {
                                    // Scale the geometry at 0.5.
                                    let transformed_point = transformation::transform_points_4x3(
                                        &scale_matrix,
                                        &mesh.vertices,
                                    );
                                    let r = camera.project_a_list_of_points(&transformed_point);
                                    // Could have been parallelized i don't know if that would have
                                    // been beneficial... not a big deal for now.
                                    for projected_point in r.iter() {
                                        buffer[projected_point.1 * screen_width
                                            + projected_point.0] = 0x00004e;
                                    }
                                }
                            }
                        }
                    }
                }

                ////////////////////////////////////////////////////////////////
                /////////////////////////////////////////////////////////////////////////////////////
                // Draw a parametric block mesh
                /////////////////////////////////////////////////////////////////////////////////////
                let origin = Vertex::new(0.0, 0.0, 0.0);
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
                    0.2,
                    0.1,
                    0.1,
                    10,
                    1,
                    1,
                )
                .to_mesh();
                /////////////////////////////////////////////////////////////////////////////////////
                /////////////////////////////////////////////////////////////////////////////////////
                let box_normals_list = m_box.extract_faces_normals_vectors();
                for (i, norm) in box_normals_list.iter().enumerate() {
                    let p1 = camera.project_maybe_outside(&norm.0);
                    let p2 = norm.0 + (norm.1 * 0.05);
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
                // TODO: find a way to compute the camera position.
                ////////////////////////////////////////////////////////////////
                // Test for an input vector.
                ////////////////////////////////////////////////////////////////
                let pt1 = Vertex::new(0.1, -0.1, 0.1);
                let pt2 = Vertex::new(0.0,0.0,0.0);
                println!("\x1b[2K\rTest position: {0:?}, Test target: {1:?}", pt1, pt2);
                let cam_position = pt1;
                let test_dir = (pt2 - pt1).normalize();
                let cam_target = pt1 + (test_dir*0.1);
                let pt1 = camera.project_maybe_outside(&cam_position);
                let pt2 = camera.project_maybe_outside(&cam_target);
                if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                    draw_aa_line_with_thickness(&mut buffer, screen_width, pt.0, pt.1,3, 0x0000FF);
                }
                ////////////////////////////////////////////////////////////////
                // Test to retrieve the camera direction.
                ////////////////////////////////////////////////////////////////
                println!("\x1b[2K\rCam position: {0:?}, Cam target: {1:?}", camera.position, camera.target);
                let mut cam_position = camera.position.to_vertex();
                cam_position.z = -cam_position.z;
                let mut cam_dir = (camera.target.to_vertex() - cam_position).normalize()*1.0;
                println!("\x1b[2K\rTest: Cam position: {0:?} (remapped), Cam target: {1:?} Length:{2:?}", cam_position, camera.target,cam_dir.magnitude());
                cam_position = camera.target.to_vertex() + -cam_dir;
                // cam_position = camera.multiply_matrix_vector(&camera.view_matrix, &cam_position);
                let pt1 = camera.project_maybe_outside(&cam_position);
                let pt2 = camera.project_maybe_outside(&camera.target.to_vertex());
                if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                    draw_aa_line_with_thickness(&mut buffer, screen_width, pt.0, pt.1, 3,0xFF0000);
                }
                ////////////////////////////////////////////////////////////////
                let mut c_d = camera.get_camera_direction();
                c_d.y = -c_d.y; 

                let c_o = camera.get_camera_origin();
                println!("\x1b[2K\rTest B: Cam c_d: {0:?}  Cam c_o: {1:?}", c_d, c_o);
                let pt1 = camera.project_maybe_outside(&c_d);
                let pt2 = camera.project_maybe_outside(&Vertex::new(0.0,0.0,0.0));
                if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                    draw_aa_line(&mut buffer, screen_width, pt.0, pt.1, 0xFF00FF);
                }
                /*

                let pt1 = -camera.multiply_matrix_vector(&camera.view_matrix,&camera.position.to_vertex());
                let pt2 = camera.multiply_matrix_vector(&camera.view_matrix,&camera.target.to_vertex());
                let remap_position = Vertex::new(  -pt1.x, -pt1.z, -pt1.y);
                println!(
                    "\x1b[2K\rCam position: {0:?}, remapped position:{1:?}, cam target: {2:?}",
                    pt1,
                    remap_position,
                    pt2,
                );
                let cam_position = pt1;
                let cam_dir = pt1 - pt2;
                let cam_target = pt1 + cam_dir;
                let pt1 = camera.project_maybe_outside(&remap_position);
                let pt2 = camera.project_maybe_outside(&cam_target);
                if let Some(pt) = clip_line(pt1, pt2, screen_width, screen_height) {
                    draw_aa_line(&mut buffer, screen_width, pt.0, pt.1, 0x0);
                }
                */
                // This is working with a regular vector. (now i just need to compute the camera position.)
                let border_edges = m_box.extract_silhouette(&test_dir);
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

                // Test triangle indices localization.
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
        runtime.push(FunctionPointer::Inverse_sqrt(inverse_sqrt));
        runtime.push(FunctionPointer::Some_operation(some_operation));
        runtime.push(FunctionPointer::Display_struct(display_struct));

        // Feed parameters from struct related to the stack order.
        for (index, fpointer) in runtime.iter().enumerate() {
            match fpointer {
                FunctionPointer::Magnetude(f) => {
                    println!("--->id:{index} fcall:{:?}", f(3.0, 4.0, 0.0));
                }
                FunctionPointer::Inverse_sqrt(f) => {
                    println!("--->id:{index} fcall:{:?}", f(81.0));
                }
                FunctionPointer::Some_operation(f) => {
                    println!("--->id:{index} fcall:{:?}", f(2, 2));
                }
                FunctionPointer::Display_struct(f) => {
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
        Inverse_sqrt(fn(value: f64) -> f64),
        Some_operation(fn(a: u8, b: u8) -> u8), // Generic have to by call contextually.
        Display_struct(fn(&Vector2d) -> String),
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
