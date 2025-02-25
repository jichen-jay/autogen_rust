use lazy_static::lazy_static;
use serde_json::from_str as json_from_str;
use std::collections::HashMap;
use std::convert::TryInto;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Mutex;

pub type MyResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

lazy_static! {
    pub static ref STORE: Mutex<HashMap<String, Tool>> = Mutex::new(HashMap::new());
}

pub struct Tool {
    pub name: String,
    pub function: Arc<dyn Fn(&[String]) -> MyResult<String> + Send + Sync>,
    pub tool_def_obj: String,
    pub arg_names: Vec<String>,
    pub arg_types: Vec<String>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("tool_def_obj", &self.tool_def_obj)
            .field("arg_names", &self.arg_names)
            .field("arg_types", &self.arg_types)
            .finish()
    }
}

impl Tool {
    pub fn run(&self, arguments_w_val: String) -> MyResult<String> {
        if arguments_w_val.trim().is_empty() {
            return (self.function)(&[]);
        }
        let parsed: serde_json::Value = serde_json::from_str(&arguments_w_val)
            .map_err(|e| format!("Failed to parse arguments JSON: {:?}", e))?;

        let arguments = parsed
            .as_object()
            .ok_or("Invalid arguments format: expected a JSON object")?;

        let mut ordered_vals = Vec::new();

        for arg_name in &self.arg_names {
            let arg_value = if let Some(args) = arguments.get("arguments") {
                if let Some(array) = args.as_array() {
                    let mut found = None;
                    for item in array {
                        if let Some(obj) = item.as_object() {
                            if let Some(value) = obj.get(arg_name) {
                                found = Some(value);
                                break;
                            }
                        }
                    }
                    found.ok_or(format!("Missing argument: {}", arg_name))?
                } else if let Some(obj) = args.as_object() {
                    obj.get(arg_name)
                        .ok_or(format!("Missing argument: {}", arg_name))?
                } else {
                    return Err("Invalid arguments format".into());
                }
            } else {
                arguments
                    .get(arg_name)
                    .ok_or(format!("Missing argument: {}", arg_name))?
            };

            let arg_str = if let Some(s) = arg_value.as_str() {
                s.to_string()
            } else {
                serde_json::to_string(arg_value).map_err(|e| {
                    format!(
                        "Failed to convert argument '{}' to string: {:?}",
                        arg_name, e
                    )
                })?
            };

            ordered_vals.push(arg_str);
        }

        (self.function)(&ordered_vals)
    }
}

pub trait TypeConverter {
    type Output;
    fn convert_from_str(s: &str) -> Result<Self::Output, String>;
}

impl TypeConverter for i8 {
    type Output = i8;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<i8>()
            .map_err(|e| format!("Conversion error for i8: {:?}", e))
    }
}

impl TypeConverter for i16 {
    type Output = i16;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<i16>()
            .map_err(|e| format!("Conversion error for i16: {:?}", e))
    }
}

impl TypeConverter for i32 {
    type Output = i32;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<i32>()
            .map_err(|e| format!("Conversion error for i32: {:?}", e))
    }
}

impl TypeConverter for i64 {
    type Output = i64;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<i64>()
            .map_err(|e| format!("Conversion error for i64: {:?}", e))
    }
}

impl TypeConverter for isize {
    type Output = isize;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<isize>()
            .map_err(|e| format!("Conversion error for isize: {:?}", e))
    }
}

impl TypeConverter for u8 {
    type Output = u8;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<u8>()
            .map_err(|e| format!("Conversion error for u8: {:?}", e))
    }
}

impl TypeConverter for u16 {
    type Output = u16;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<u16>()
            .map_err(|e| format!("Conversion error for u16: {:?}", e))
    }
}

impl TypeConverter for u32 {
    type Output = u32;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<u32>()
            .map_err(|e| format!("Conversion error for u32: {:?}", e))
    }
}

impl TypeConverter for u64 {
    type Output = u64;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<u64>()
            .map_err(|e| format!("Conversion error for u64: {:?}", e))
    }
}

impl TypeConverter for usize {
    type Output = usize;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<usize>()
            .map_err(|e| format!("Conversion error for usize: {:?}", e))
    }
}

impl TypeConverter for f32 {
    type Output = f32;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<f32>()
            .map_err(|e| format!("Conversion error for f32: {:?}", e))
    }
}

impl TypeConverter for f64 {
    type Output = f64;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<f64>()
            .map_err(|e| format!("Conversion error for f64: {:?}", e))
    }
}

impl TypeConverter for char {
    type Output = char;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        let mut chars = s.chars();
        if let Some(c) = chars.next() {
            if chars.next().is_some() {
                Err(format!("Expected a single character, got '{}'", s))
            } else {
                Ok(c)
            }
        } else {
            Err("Empty string provided for char".to_string())
        }
    }
}

impl TypeConverter for bool {
    type Output = bool;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        s.parse::<bool>()
            .map_err(|e| format!("Conversion error for bool: {:?}", e))
    }
}

impl TypeConverter for String {
    type Output = String;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        Ok(s.to_string())
    }
}

impl TypeConverter for () {
    type Output = ();
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        if s.trim().is_empty() || s.trim() == "()" {
            Ok(())
        } else {
            Err(format!(
                "Expected unit type (empty input or '()'), got '{}'",
                s
            ))
        }
    }
}

impl<T> TypeConverter for Vec<T>
where
    T: FromStr + serde::de::DeserializeOwned,
    T::Err: std::fmt::Debug,
{
    type Output = Vec<T>;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        if let Ok(vec) = json_from_str::<Vec<T>>(s) {
            return Ok(vec);
        }
        s.split(',')
            .map(|item| item.trim())
            .filter(|item| !item.is_empty())
            .map(|item| {
                item.parse::<T>().map_err(|e| {
                    format!(
                        "Failed to parse '{}' as {}: {:?}",
                        item,
                        std::any::type_name::<T>(),
                        e
                    )
                })
            })
            .collect()
    }
}

impl<K, V> TypeConverter for HashMap<K, V>
where
    K: FromStr + std::cmp::Eq + std::hash::Hash + serde::de::DeserializeOwned,
    V: FromStr + serde::de::DeserializeOwned,
    K::Err: std::fmt::Debug,
    V::Err: std::fmt::Debug,
{
    type Output = HashMap<K, V>;
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        if let Ok(map) = json_from_str::<HashMap<K, V>>(s) {
            return Ok(map);
        }
        s.split(',')
            .map(|pair| {
                let mut parts = pair.split(':').map(|p| p.trim());
                match (parts.next(), parts.next()) {
                    (Some(key), Some(value)) => {
                        let k = key
                            .parse::<K>()
                            .map_err(|e| format!("Failed to parse key '{}': {:?}", key, e))?;
                        let v = value
                            .parse::<V>()
                            .map_err(|e| format!("Failed to parse value '{}': {:?}", value, e))?;
                        Ok((k, v))
                    }
                    _ => Err(format!("Invalid key-value pair: {}", pair)),
                }
            })
            .collect()
    }
}

impl<T, const N: usize> TypeConverter for [T; N]
where
    T: FromStr + serde::de::DeserializeOwned,
    T::Err: std::fmt::Debug,
{
    type Output = [T; N];
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        let vec: Vec<T> = <Vec<T> as TypeConverter>::convert_from_str(s)?;
        if vec.len() != N {
            return Err(format!(
                "Expected array of length {}, but got {}",
                N,
                vec.len()
            ));
        }
        vec.try_into()
            .map_err(|_| format!("Failed to convert Vec to array of length {}", N))
    }
}

impl<A, B> TypeConverter for (A, B)
where
    A: FromStr + serde::de::DeserializeOwned,
    A::Err: std::fmt::Debug,
    B: FromStr + serde::de::DeserializeOwned,
    B::Err: std::fmt::Debug,
{
    type Output = (A, B);
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        // First try JSON array conversion.
        if let Ok(tuple) = serde_json::from_str::<(A, B)>(s) {
            return Ok(tuple);
        }
        // Otherwise assume comma-separated format.
        let parts: Vec<&str> = s.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(format!(
                "Expected 2 elements for tuple (A, B), found {}",
                parts.len()
            ));
        }
        let a = parts[0]
            .parse::<A>()
            .map_err(|e| format!("Conversion error for first element: {:?}", e))?;
        let b = parts[1]
            .parse::<B>()
            .map_err(|e| format!("Conversion error for second element: {:?}", e))?;
        Ok((a, b))
    }
}

// Tuple of three elements.
impl<A, B, C> TypeConverter for (A, B, C)
where
    A: FromStr + serde::de::DeserializeOwned,
    A::Err: std::fmt::Debug,
    B: FromStr + serde::de::DeserializeOwned,
    B::Err: std::fmt::Debug,
    C: FromStr + serde::de::DeserializeOwned,
    C::Err: std::fmt::Debug,
{
    type Output = (A, B, C);
    fn convert_from_str(s: &str) -> Result<Self::Output, String> {
        if let Ok(tuple) = serde_json::from_str::<(A, B, C)>(s) {
            return Ok(tuple);
        }
        let parts: Vec<&str> = s.split(',').map(|s| s.trim()).collect();
        if parts.len() != 3 {
            return Err(format!(
                "Expected 3 elements for tuple (A, B, C), found {}",
                parts.len()
            ));
        }
        let a = parts[0]
            .parse::<A>()
            .map_err(|e| format!("Conversion error for first element: {:?}", e))?;
        let b = parts[1]
            .parse::<B>()
            .map_err(|e| format!("Conversion error for second element: {:?}", e))?;
        let c = parts[2]
            .parse::<C>()
            .map_err(|e| format!("Conversion error for third element: {:?}", e))?;
        Ok((a, b, c))
    }
}

