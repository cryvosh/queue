use super::TimingData;
use dioxus::prelude::*;

#[component]
pub fn Canvas(timing_data: Signal<TimingData>) -> Element {
    let canvas_id = "queues-canvas";

    #[cfg(feature = "web")]
    use_effect(move || {
        let canvas_id = canvas_id.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(err) = crate::web_runtime::start_renderer(&canvas_id, timing_data).await {
                web_sys::console::error_1(&format!("Renderer error: {}", err).into());
            }
        });
    });

    rsx! {
        div {
            id: "canvas-container",
            style: "margin: 0; width: 100vw; height: 100vh; padding: 0; position: relative;",
            canvas {
                id: canvas_id,
                style: "display: block; width: 100%; height: 100%; background: #000;",
                "Your browser does not support the canvas element."
            }
        }
    }
}
