use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Expr, ItemFn};

#[proc_macro_attribute]
pub fn create_tool_with_function(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    let attr_expr = parse_macro_input!(attr as Expr);

    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();

    let register_fn_name = format_ident!("register_tool_{}", fn_name_str);

    let inputs = &input_fn.sig.inputs;
    let mut arg_names = Vec::new();
    let mut arg_type_tokens = Vec::new();

    for arg in inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let arg_name = &pat_ident.ident;
                arg_names.push(quote! { #arg_name });
            }
            let arg_type = &*pat_type.ty;
            arg_type_tokens.push(arg_type.clone());
        }
    }

    let gen = quote! {
                #input_fn

                #[ctor]
                fn #register_fn_name() {
                    let arg_names = vec![#(stringify!(#arg_names).to_string()),*];
                    let arg_types = vec![#(stringify!(#arg_type_tokens).to_string()),*];

                    let func = {
                        use std::sync::Arc;
                        let func = Arc::new(move |args: &[SupportedType]| -> MyResult<String> {
                            let parsers = get_parsers();

                            let mut iter = args.iter();
                            #(
                                let arg_type = stringify!(#arg_type_tokens);
                                let #arg_names = {
                                    let arg = iter.next().ok_or("Not enough arguments")?.clone();
                                    let parser = parsers.get(arg_type)
                                        .ok_or(format!("Parser not found for type {}", arg_type))?;
                                    let any_val = parser(arg)?;
                                    let val = any_val.downcast::<#arg_type_tokens>()
                                        .map_err(|_| "Type mismatch")?;
                                    *val
                                };
                            )*

                            #fn_name(#(#arg_names),*)
                        }) as Arc<dyn Fn(&[SupportedType]) -> MyResult<String> + Send + Sync>;
                        func
                    };

                    let tool_def_obj: String = (#attr_expr).to_string();

    use serde::Deserialize;

        #[derive(Deserialize)]
        struct ToolMeta {
            name: String,
        }

        let json_name = match serde_json::from_str::<ToolMeta>(&tool_def_obj) {
            Ok(tool_meta) => tool_meta.name,
            Err(_) => "Unknown".to_string(),
        };
                    let fn_name_str = stringify!(#fn_name);

                    if fn_name_str != json_name {
                        panic!(
                            "Function name '{}' does not match 'name' field '{}' in tool definition",
                            fn_name_str, json_name
                        );
                    }

                    let tool = Tool {
                        name: (#fn_name_str).to_string(),
                        function: func,
                        tool_def_obj: tool_def_obj,
                        arg_names: arg_names,
                        arg_types: arg_types,
                    };

                    {
                        STORE.lock().unwrap().insert(tool.name.clone(), tool.clone());
                    }
                }
            };

    gen.into()
}
