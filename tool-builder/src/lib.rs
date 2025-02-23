extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Expr, ItemFn};
// use tool_builder_core;

#[proc_macro_attribute]
pub fn create_tool_with_function(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the attribute (tool metadata) and function item.
    let input_fn = parse_macro_input!(item as ItemFn);
    let attr_expr = parse_macro_input!(attr as Expr);
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let register_fn_name = format_ident!("register_tool_{}", fn_name_str);

    // Extract argument names and types.
    let mut arg_names = Vec::new();
    let mut arg_types = Vec::new();
    for input in &input_fn.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                arg_names.push(pat_ident.ident.clone());
            }
            arg_types.push((*pat_type.ty).clone());
        }
    }

    // Generate code that creates a wrapper function converting string arguments.
    let gen = quote! {
        #input_fn

        #[ctor::ctor]
        fn #register_fn_name() {
            let arg_names = vec![#(stringify!(#arg_names).to_string()),*];
            let arg_types = vec![#(stringify!(#arg_types).to_string()),*];

            use std::sync::Arc;
            let func = Arc::new(move |args: &[String]| -> MyResult<String> {
                let mut iter = args.iter();
                #(
                    let #arg_names = {
                        let arg_str = iter.next().ok_or_else(||
                            format!("Missing argument for parameter '{}'", stringify!(#arg_names))
                        )?;
                        <#arg_types as TypeConverter>::convert_from_str(arg_str)
                           .map_err(|e| format!("Failed to convert argument '{}': {}", stringify!(#arg_names), e))?
                    };
                )*
                #fn_name( #(#arg_names),* )
            }) as Arc<dyn Fn(&[String]) -> MyResult<String> + Send + Sync>;

            let tool_def_obj: String = (#attr_expr).to_string();

            #[derive(serde::Deserialize)]
            struct ToolMeta {
                name: String,
                description: Option<String>,
                parameters: Option<serde_json::Value>,
            }
            let tool_meta = serde_json::from_str::<ToolMeta>(&tool_def_obj)
                .expect("Failed to parse tool definition");
            if #fn_name_str != tool_meta.name {
                panic!(
                    "Function name '{}' does not match the 'name' field '{}' in the tool definition",
                    #fn_name_str, tool_meta.name
                );
            }
            let tool = Tool {
                name: #fn_name_str.to_string(),
                function: func,
                tool_def_obj,
                arg_names,
                arg_types,
            };
            STORE.lock().unwrap().insert(tool.name.clone(), tool);
        }
    };

    gen.into()
}
