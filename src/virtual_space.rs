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
//  - Before sending to display pipe line iter from LayerVisibility object and
//    apply related parmeter in function.

use crate::display_pipe_line::rendering_object::{Mesh, Vertex};
use crate::display_pipe_line::visualization_v3::coloring::*;
use crate::rust3d::geometry::*;
use crate::rust3d::transformation::*;
use chrono::Local;
use std::fmt;
use std::sync::Arc;
////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub struct Virtual_space {
    pub project_name: String,
    pub file_path: Option<String>,
    pub unit_scale: Unit_scale,
    pub display: Display_config,
    object_list: Vec<Object3d>, // array of Object3d.
    pub layers: Vec<LayerVisibility>,
}
impl Virtual_space {
    /// Constructor of the main class.
    /// define principal options for the
    /// user interactive system.
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
        }
    }
    /// Add a new object on the stack.
    pub fn add_obj(&mut self, object: Object3d) {
        self.object_list.push(object);
    }
    /// Replace a displayable on same location object..
    pub fn replace_displayable_in_place(
        &mut self,
        virtual_space_obj_index: usize,
        new_displayable_data: Displayable,
    ) {
        // Get the object to be modified
        let object = &mut self.object_list[virtual_space_obj_index];

        // Push the current data to the undo stack if it exists
        if let Some(current_data) = object.data.clone() {
            Arc::make_mut(&mut object.undo_stack).push((*current_data).clone());
        }

        // Replace the current `data` with the new `Displayable`
        object.data = Some(Arc::new(new_displayable_data));
    }
    pub fn undo_displayable_in_place(&mut self, virtual_space_obj_index: usize) {
        let object = &mut self.object_list[virtual_space_obj_index];

        if let Some(undo_data) = Arc::make_mut(&mut object.undo_stack).pop() {
            if let Some(current_data) = object.data.clone() {
                Arc::make_mut(&mut object.redo_stack).push((*current_data).clone());
            }

            // Restore the previous state
            object.data = Some(Arc::new(undo_data));
        } else {
            eprintln!("Nothing to undo... no change has occurred.");
        }
    }
    pub fn redo_displayable_in_place(&mut self, virtual_space_obj_index: usize) {
        let object = &mut self.object_list[virtual_space_obj_index];

        if let Some(redo_data) = Arc::make_mut(&mut object.redo_stack).pop() {
            if let Some(current_data) = object.data.clone() {
                Arc::make_mut(&mut object.undo_stack).push((*current_data).clone());
            }

            // Restore the next state
            object.data = Some(Arc::new(redo_data));
        } else {
            eprintln!("Nothing to redo... no change has occurred.");
        }
    }
    /// Clean the stack of Empty elements.
    pub fn clean_stack(&mut self) {
        self.object_list.retain(|obj| obj.data.is_none());
    }

    // Other methods remain unchanged...

    /// Add a layer visibility.
    pub fn add_layer(&mut self, layer: LayerVisibility) {
        self.layers.push(layer);
    }

    /// Toggle visibility for a specific layer.
    pub fn toggle_layer_visibility(&mut self, layer_index: usize) {
        if let Some(layer) = self.layers.get_mut(layer_index) {
            layer.visibility = !layer.visibility;
        }
    }

    /// Lock or unlock a specific layer.
    pub fn toggle_layer_lock(&mut self, layer_index: usize) {
        if let Some(layer) = self.layers.get_mut(layer_index) {
            layer.lock = !layer.lock;
        }
    }
}
impl fmt::Display for Virtual_space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let project_name = format!("{}", self.project_name);
        let path = if let Some(path) = &self.file_path {
            format!("{0}", path)
        } else {
            format!("None")
        };
        let unit_scale = self.unit_scale.to_string();
        let display_config = format!(
            "height:{}, width:{}, ratio:{}, raytrace enabled:({}).",
            self.display.display_resolution_height,
            self.display.display_resolution_width,
            self.display.display_ratio,
            self.display.raytrace
        );
        let mut obj_list = String::new();
        obj_list.push_str(&format!("Contain ({}) Object3d: ", self.object_list.len()));
        for i in 0..self.object_list.len() {
            let data_display = match &self.object_list[i].data {
                Some(disp) => disp.to_string(),
                None => "None".to_string(),
            };
            obj_list.push_str(data_display.as_str());
            obj_list.push_str(format!(" (index: {0}), ", i).as_str());
        }
        let mut layers_str = String::new();
        if self.layers.len() == 0{
            layers_str.push_str("No layers created yet.");
        }else{
        for (i, layer) in self.layers.iter().enumerate() {
            layers_str.push_str(&format!(
                "Layer {}: Visible: {}, Locked: {}, Color: {:?}\n",
                i, layer.visibility, layer.lock, layer.color
            ));
            }
        }
        write!(
            f,
            "Virtual Space:
                Project name: '{0}'
                File path: {1}
                Unit scale: {2}
                Display Pipeline Config: {3}
                Object3d List: {4}
                layers List: {5}
                ",
            project_name, path, unit_scale, display_config, obj_list,layers_str
        )
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Oject System.
#[derive(Debug)]
pub struct Object3d {
    pub origin: CPlane,
    pub data: Option<Arc<Displayable>>, // if object is removed position is kept
    undo_stack: Arc<Vec<Displayable>>,
    redo_stack: Arc<Vec<Displayable>>,
    pub local_scale_ratio: f64,
    pub id: u64,
    pub last_change_date: String,
}

impl Object3d {
    /// Create an object ready to be stacked
    pub fn new(
        id: u64,
        data: Option<Arc<Displayable>>,
        origin: CPlane,
        local_scale_ratio: f64,
    ) -> Self {
        Self {
            id,
            data,
            origin,
            local_scale_ratio,
            last_change_date: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            undo_stack: Arc::new(Vec::new()),
            redo_stack: Arc::new(Vec::new()),
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
            origin: self.origin.clone(),              // Deep copy origin
            data: self.data.clone(),                  // Clone Arc (pointer)
            undo_stack: Arc::clone(&self.undo_stack), // Clone Arc (pointer)
            redo_stack: Arc::clone(&self.redo_stack), // Clone Arc (pointer)
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
            Some(disp) => disp.to_string(),
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
    pub display_ratio: f64,
    pub camera_position: [[f64; 3]; 4], // special matrix format optimized
    // for visualization system.
    pub raytrace: bool,
}
impl Display_config {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            display_resolution_height: height,
            display_resolution_width: width,
            display_ratio: (height as f64 / width as f64),
            // create an identity matrix
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
// a nested list will send data to display pipeline from there.
#[derive(Debug)]
struct LayerVisibility {
    object3d_list_index: Vec<usize>,
    visibility: bool,
    color: Color,
    lock: bool,
}

////////////////////////////////////////////////////////////////////////////////
/*
 * this is a template prototype of the scripted runtime
 * from json Deserialization.
 */
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
