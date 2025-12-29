use crate::components::TimingData;
use crate::renderer::Renderer;
use crate::server::fetch_shaders;
use dioxus::prelude::{Signal, WritableExt};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

const RESOLUTION_SCALE: f32 = 1.0;

async fn fetch_shaders_from_server() -> Result<std::collections::HashMap<String, String>, String> {
    fetch_shaders()
        .await
        .map_err(|e| format!("Server function error: {:?}", e))
}

fn get_canvas(canvas_id: &str) -> Result<web_sys::HtmlCanvasElement, String> {
    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| "Not a canvas")?;
    Ok(canvas)
}

fn update_canvas_dimensions(canvas: &web_sys::HtmlCanvasElement) -> Option<(u32, u32)> {
    let container = canvas.parent_element()?;
    let window = web_sys::window()?;

    let container_width = container.client_width() as f32;
    let container_height = container.client_height() as f32;
    let device_pixel_ratio = window.device_pixel_ratio() as f32;

    if container_width <= 0.0 || container_height <= 0.0 {
        return None;
    }

    let physical_width = (container_width * device_pixel_ratio * RESOLUTION_SCALE).round() as u32;
    let physical_height = (container_height * device_pixel_ratio * RESOLUTION_SCALE).round() as u32;

    if canvas.width() != physical_width {
        canvas.set_width(physical_width);
    }
    if canvas.height() != physical_height {
        canvas.set_height(physical_height);
    }

    Some((physical_width, physical_height))
}

pub async fn start_renderer(
    canvas_id: &str,
    mut timing_data: Signal<TimingData>,
) -> Result<(), String> {
    let canvas = get_canvas(canvas_id)?;

    if let Some((w, h)) = update_canvas_dimensions(&canvas) {
        log::info!("Initial canvas dimensions: {}x{}", w, h);
    }

    let mut renderer = Renderer::new(&canvas).await?;

    renderer.initialize_buffers();

    let shaders = fetch_shaders_from_server().await?;
    let compiled = Renderer::compile_shaders_async(&renderer.device, &shaders).await?;
    renderer.install_compiled_shaders(compiled);
    renderer.build_pipelines()?;

    let renderer = Rc::new(RefCell::new(renderer));
    let canvas_clone = canvas.clone();
    let reload_requested = Rc::new(Cell::new(false));
    let reload_in_flight = Rc::new(Cell::new(false));
    let last_frame_time = Rc::new(Cell::new(0.0f64));

    setup_reload_key_handler(reload_requested.clone());

    let raf_cb: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let raf_cb_clone = raf_cb.clone();

    let cb = Closure::wrap(Box::new(move |time_ms: f64| {
        let canvas = canvas_clone.clone();
        let renderer_clone = renderer.clone();
        
        let (width, height) = match update_canvas_dimensions(&canvas) {
            Some(dims) => dims,
            None => (canvas.width().max(1), canvas.height().max(1)),
        };
        
        let time = (time_ms / 1000.0) as f32;

        let prev_time = last_frame_time.get();
        let frame_time = if prev_time > 0.0 {
            time_ms - prev_time
        } else {
            0.0
        };
        last_frame_time.set(time_ms);
        let start_cpu = web_sys::window().unwrap().performance().unwrap().now();

        if reload_requested.get() && !reload_in_flight.get() {
            reload_requested.set(false);

            if let Ok(r) = renderer.try_borrow() {
                let device = r.device.clone();
                drop(r);

                reload_in_flight.set(true);
                let renderer_for_reload = renderer.clone();
                let reload_in_flight_clone = reload_in_flight.clone();

                wasm_bindgen_futures::spawn_local(async move {
                    log::info!("Q pressed: reloading shaders...");

                    let shaders = match fetch_shaders_from_server().await {
                        Ok(s) => s,
                        Err(e) => {
                            web_sys::console::error_1(
                                &format!("Shader fetch failed: {}", e).into(),
                            );
                            reload_in_flight_clone.set(false);
                            return;
                        }
                    };

                    let compiled = match Renderer::compile_shaders_async(&device, &shaders).await {
                        Ok(c) => c,
                        Err(e) => {
                            web_sys::console::error_1(
                                &format!("Shader compile failed: {}", e).into(),
                            );
                            reload_in_flight_clone.set(false);
                            return;
                        }
                    };

                    match renderer_for_reload.try_borrow_mut() {
                        Ok(mut r) => {
                            r.install_compiled_shaders(compiled);
                            match r.build_pipelines() {
                                Ok(()) => log::info!("✓ Shaders reloaded"),
                                Err(e) => web_sys::console::error_1(
                                    &format!("Pipeline rebuild failed: {}", e).into(),
                                ),
                            }
                        }
                        Err(_) => {
                            web_sys::console::error_1(
                                &"Shader install failed (renderer busy)".into(),
                            );
                        }
                    }

                    reload_in_flight_clone.set(false);
                });
            } else {
                reload_requested.set(true);
            }
        }

        if let Ok(mut renderer) = renderer.try_borrow_mut() {
            if let Err(e) = renderer.render(time, width, height) {
                web_sys::console::error_1(&format!("Render error: {}", e).into());
            }

            let end_cpu = web_sys::window().unwrap().performance().unwrap().now();
            let cpu_ms = end_cpu - start_cpu;

            let results = renderer.get_timing_results();
            let total_gpu_ms: f64 = results.iter().map(|r| r.duration_ms).sum();
            let entries = results
                .into_iter()
                .map(|r| {
                    let percentage = if total_gpu_ms > 0.0 {
                        (r.duration_ms / total_gpu_ms * 100.0).min(100.0)
                    } else {
                        0.0
                    };
                    crate::components::TimingEntry {
                        name: r.name,
                        duration_ms: r.duration_ms,
                        percentage,
                    }
                })
                .collect::<Vec<_>>();

            timing_data.set(TimingData {
                cpu_ms,
                frame_time_ms: frame_time,
                entries,
                total_ms: total_gpu_ms,
            });
        }

        if let Some(cb) = raf_cb_clone.borrow().as_ref() {
            let window = web_sys::window().unwrap();
            window
                .request_animation_frame(cb.as_ref().unchecked_ref())
                .ok();
        }
    }) as Box<dyn FnMut(f64)>);

    let window = web_sys::window().unwrap();
    window
        .request_animation_frame(cb.as_ref().unchecked_ref())
        .ok();
    *raf_cb.borrow_mut() = Some(cb);

    Ok(())
}

fn setup_reload_key_handler(reload_requested: Rc<Cell<bool>>) {
    let target: web_sys::EventTarget = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .dyn_into()
        .unwrap();

    let cb = Closure::wrap(Box::new(move |event: web_sys::Event| {
        if let Ok(ke) = event.dyn_into::<web_sys::KeyboardEvent>() {
            if ke.code() == "KeyQ" {
                reload_requested.set(true);
                ke.prevent_default();
            }
        }
    }) as Box<dyn FnMut(web_sys::Event)>);

    target
        .add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())
        .ok();

    cb.forget();
}
