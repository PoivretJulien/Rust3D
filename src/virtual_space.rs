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

use crate::display_pipe_line::rendering_object::{Mesh, Vertex};
use crate::display_pipe_line::visualization_v3::coloring::*;
use crate::rust3d::geometry::*;
use crate::rust3d::transformation::*;
use chrono::Local;
////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
pub struct Virtual_space {
    pub project_name: String,
    pub file_path: Option<String>,
    pub unit_scale: Unit_scale,
    pub display: Display_config,
    object_list: Vec<Object3d>, // array of Object3d.
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
        }
    }
    /// Add a new object on the stack.
    pub fn add_obj(&mut self, object: Object3d) {
        self.object_list.push(object);
    }
    /// Replace a displayable on same location object..
    pub fn replace_displayable_in_place(
        &mut self,
        index: usize,
        new_displayable_data: Displayable,
    ) {
        // Check if there is an object and then proceed to backup and replacement.
        // if not simply replace the object with no backup.
        if let Some(data) = self.object_list[index].data.clone() {
            self.object_list[index].undo_stack.push(data); // Backup a copy on undo stack.
            self.object_list[index].data = Some(new_displayable_data);
        } else {
            self.object_list[index].data = Some(new_displayable_data);
        }
    }
    /// Recover previous change.
    pub fn undo_displayable_in_place(&mut self, index: usize) {
        if self.object_list[index].undo_stack.len() > 0 {
            if let Some(data) = self.object_list[index].data.clone(){
                self.object_list[index].redo_stack.push(data); // place element in redo stack.
                // put back in place undo element.
                self.object_list[index].data = self.object_list[index].undo_stack.pop();
            }
        } else {
            eprintln!("Nothing to undo... no change has occurred.")
        }
    }
    /// Recover previous change.
    pub fn redo_displayable_in_place(&mut self, index: usize) {
        if self.object_list[index].redo_stack.len() > 0 {
            if let Some(data) = self.object_list[index].data.clone() {
                self.object_list[index].undo_stack.push(data);
                self.object_list[index].data = self.object_list[index].redo_stack.pop();
            }
        } else {
            eprintln!("Nothing to undo... no change has occurred.")
        }
    }
    /// Clean the stack of Empty elements.
    pub fn clean_stack(&mut self) {
        self.object_list.retain(|obj| obj.data.is_none());
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Oject System.
#[derive(Debug)]
pub struct Object3d {
    pub origin: CPlane,
    pub data: Option<Displayable>, // if object is removed position is kept
    undo_stack: Vec<Displayable>,
    redo_stack: Vec<Displayable>,
    pub local_scale_ratio: f64,
    pub id: u64,
    pub last_change_date: String,
}
impl Object3d {
    /// Create an object ready to be stacked
    pub fn new(id: u64, data: Option<Displayable>, origin: CPlane, local_scale_ratio: f64) -> Self {
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
/// every thing that can be displayed for now
/// Curve are basically a set of points.
#[derive(Clone, Debug)]
pub enum Displayable {
    Point3d(Vec<Point3d>),
    Vector3d(Vec<Vector3d>),
    Vertex(Vec<Vertex>),
    Mesh(Mesh),
}
// metric or imperial system reference.
#[derive(Clone, Debug)]
pub enum Unit_scale {
    Minimeters,
    Centimeters,
    Meters,
    Inch,
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
