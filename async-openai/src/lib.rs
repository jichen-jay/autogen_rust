//! Rust library for OpenAI
//!
//! ## Creating client
//!
//! ```
//! use async_openai::{Client, config::OpenAIConfig};
//!
//! // Create a OpenAI client with api key from env var OPENAI_API_KEY and default base url.
//! let client = Client::new();
//!
//! // Above is shortcut for
//! let config = OpenAIConfig::default();
//! let client = Client::with_config(config);
//!
//! // OR use API key from different source and a non default organization
//! let api_key = "sk-..."; // This secret could be from a file, or environment variable.
//! let config = OpenAIConfig::new()
//!     .with_api_key(api_key)
//!     .with_org_id("the-continental");
//!
//! let client = Client::with_config(config);
//!
//! // Use custom reqwest client
//! let http_client = reqwest::ClientBuilder::new().user_agent("async-openai-").build().unwrap();
//! let client = Client::new().with_http_client(http_client);
//! ```
//!
//! ## Microsoft Azure Endpoints
//!
//! ```
//! use async_openai::{Client, config::AzureConfig};
//!
//! let config = AzureConfig::new()
//!     .with_api_base("https://my-resource-name.openai.azure.com")
//!     .with_api_version("2023-03-15-preview")
//!     .with_deployment_id("deployment-id")
//!     .with_api_key("...");
//!
//! let client = Client::with_config(config);
//!
//! // Note that `async-openai-` only implements OpenAI spec
//! // and doesn't maintain parity with the spec of Azure OpenAI service.
//!
//! ```
//!
//! ## Making requests
//!
//!```
//!# tokio_test::block_on(async {
//!
//! use async_openai::{Client, types::{CreateCompletionRequestArgs}};
//!
//! // Create client
//! let client = Client::new();
//!
//! // Create request using builder pattern
//! // Every request struct has companion builder struct with same name + Args suffix
//! let request = CreateCompletionRequestArgs::default()
//!     .model("gpt-3.5-turbo-instruct")
//!     .prompt("Tell me the recipe of alfredo pasta")
//!     .max_tokens(40_u32)
//!     .build()
//!     .unwrap();
//!
//! // Call API
//! let response = client
//!     .completions()      // Get the API "group" (completions, images, etc.) from the client
//!     .create(request)    // Make the API call in that "group"
//!     .await
//!     .unwrap();
//!
//! println!("{}", response.choices.first().unwrap().text);
//! # });
//!```
//!
//! ## Examples
//! For full working examples for all supported features see [examples](https://github.com/64bit/async-openai/tree/main/examples) directory in the repository.
//!

#![allow(warnings, deprecated)]

mod assistant_files;
mod assistants;
mod audio;
mod batches;
mod chat;
mod client;
mod completion;
pub mod config;
mod download;
mod embedding;
pub mod error;
mod file;
mod fine_tuning;
mod image;
mod message_files;
mod messages;
mod model;
mod moderation;
mod runs;
mod steps;
mod threads;
pub mod types;
mod util;
mod vector_store_file_batches;
mod vector_store_files;
mod vector_stores;

pub use assistant_files::AssistantFiles;
pub use assistants::Assistants;
pub use audio::Audio;
pub use batches::Batches;
pub use chat::Chat;
pub use client::Client;
pub use completion::Completions;
pub use embedding::Embeddings;
pub use file::Files;
pub use fine_tuning::FineTuning;
pub use image::Images;
pub use message_files::MessageFiles;
pub use messages::Messages;
pub use model::Models;
pub use moderation::Moderations;
pub use runs::Runs;
pub use steps::Steps;
pub use threads::Threads;
pub use vector_store_file_batches::VectorStoreFileBatches;
pub use vector_store_files::VectorStoreFiles;
pub use vector_stores::VectorStores;
