use queues::App;

fn main() {
    #[cfg(not(feature = "server"))]
    {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Info);
    }
    dioxus::launch(App);
}
