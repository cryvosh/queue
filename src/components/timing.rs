//! Timing display components for performance visualization.

use dioxus::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct TimingEntry {
    pub name: String,
    pub duration_ms: f64,
    pub percentage: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimingData {
    pub cpu_ms: f64,
    pub frame_time_ms: f64,
    pub entries: Vec<TimingEntry>,
    pub total_ms: f64,
}

#[component]
pub fn TimingDisplay(timing_data: Signal<TimingData>) -> Element {
    let current_timing_data = timing_data.read();
    let show_ui = !current_timing_data.entries.is_empty()
        || current_timing_data.frame_time_ms > 0.0
        || current_timing_data.cpu_ms > 0.0;

    if !show_ui {
        return rsx! { div { style: "display: none;" } };
    }

    let fps = if current_timing_data.frame_time_ms > 0.0 {
        1000.0 / current_timing_data.frame_time_ms
    } else {
        0.0
    };
    let mut collapsed = use_signal(|| false);

    rsx! {
        div {
            class: "glassy-panel timing-panel",
            style: "position: absolute; top: 10px; left: 10px; z-index: 100; color: white; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 8px; font-family: monospace;",
            div {
                class: "timing-header-row timing-collapsible",
                style: "cursor: pointer; font-weight: bold; margin-bottom: 5px;",
                onclick: move |_| {
                    let cur = *collapsed.read();
                    collapsed.set(!cur);
                },
                span { "Timing" }
            }
            {
                if !*collapsed.read() {
                    rsx! {
                        div { class: "timing-header-row", "FPS: {fps:.1}" }
                        div { class: "timing-header-row", "Frame: {current_timing_data.frame_time_ms:.2}ms" }
                        div { class: "timing-header-row", "CPU: {current_timing_data.cpu_ms:.2}ms" }
                        div { class: "timing-header-row", "GPU: {current_timing_data.total_ms:.2}ms" }

                        div { class: "timing-gpu-breakdown",
                            style: "margin-top: 10px;",
                            {
                                if !current_timing_data.entries.is_empty() {
                                    rsx! {
                                        for entry in &current_timing_data.entries {
                                            div {
                                                key: "{entry.name}",
                                                class: "timing-entry",
                                                style: "margin-bottom: 5px;",
                                                div { class: "timing-entry-details",
                                                    style: "display: flex; justify-content: space-between;",
                                                    div { class: "timing-entry-name", "{entry.name}" }
                                                    div { class: "timing-entry-value", "{entry.duration_ms:.2}ms ({entry.percentage:.1}%)" }
                                                }
                                                div { class: "timing-progress-track",
                                                    style: "height: 4px; background: rgba(255,255,255,0.1); width: 100%;",
                                                    div {
                                                        class: "timing-progress-fill",
                                                        style: "height: 100%; background: #4af; width: {entry.percentage}%;"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    rsx! { div { class: "timing-no-data", "No GPU trace data." } }
                                }
                            }
                        }
                    }
                } else {
                    rsx! {}
                }
            }
        }
    }
}
