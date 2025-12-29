use dioxus::prelude::*;
use std::collections::HashMap;

#[cfg(feature = "server")]
use std::fs;
#[cfg(feature = "server")]
use std::path::Path;

#[get("/api/queues/shaders")]
pub async fn fetch_shaders() -> Result<HashMap<String, String>> {
    #[cfg(feature = "server")]
    {
        let shaders_dir = Path::new("shaders");
        let mut shaders = HashMap::new();

        fn read_dir_recursive(
            dir: &Path,
            _root: &Path,
            shaders: &mut HashMap<String, String>,
        ) -> std::io::Result<()> {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    read_dir_recursive(&path, _root, shaders)?;
                } else if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                        if ext == "wgsl" {
                            let filename = path.file_name().unwrap().to_str().unwrap().to_string();
                            let contents = fs::read_to_string(&path)?;
                            shaders.insert(filename, contents);
                        }
                    }
                }
            }
            Ok(())
        }

        read_dir_recursive(shaders_dir, shaders_dir, &mut shaders)?;
        Ok(shaders)
    }
    #[cfg(not(feature = "server"))]
    {
        Err(dioxus::fullstack::prelude::ServerFnError::ServerError(
            "Server function not available on client".to_string(),
        ))
    }
}
