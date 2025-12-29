pub mod components;
pub mod renderer;

#[cfg(any(feature = "web", feature = "server"))]
pub mod server;

#[cfg(feature = "web")]
pub mod web_runtime;

pub use components::{App, TimingData, TimingDisplay, TimingEntry};
pub use renderer::Renderer;

#[cfg(any(feature = "web", feature = "server"))]
pub use server::fetch_shaders;

#[cfg(feature = "web")]
pub use web_runtime::start_renderer;
