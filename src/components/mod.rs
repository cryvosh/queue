mod canvas;
mod timing;

pub use canvas::Canvas;
pub use timing::{TimingData, TimingDisplay, TimingEntry};

use dioxus::prelude::*;

const MAIN_CSS: Asset = asset!("/assets/main.css");

#[component]
pub fn App() -> Element {
    let timing_data = use_signal(TimingData::default);

    rsx! {
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        Canvas { timing_data }
        TimingDisplay { timing_data }
    }
}
