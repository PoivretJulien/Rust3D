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

use crate::display_pipe_line::redering_object::Mesh;
use crate::display_pipe_line::visualization_v3::coloring::*;
use crate::rust3d::geometry::*;
use crate::rust3d::transformation::*;
////////////////////////////////////////////////////////////////////////////////

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
        let result :Result<String, serde_json::Error>= serde_json::from_str(json_data);
        match result {
            Ok(person) => println!("Deserialized struct: {:?}", person),
            Err(e) => eprintln!("Error deserializing JSON: {}", e),
        }
        println!("Deserialized struct: {:?}", person);
    }
}
