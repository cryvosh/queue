#![cfg_attr(not(target_arch = "wasm32"), allow(unused_imports, dead_code))]

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use wgpu_utils::{
    buffer_entry, compile_shaders_default, BufferAccessMode::*, CompiledShader, Resources,
    TimingSystem,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Uniforms {
    pub time: f32,
    pub frame: u32,
    pub resolution: [u32; 2],
    pub _pad2: [u32; 4],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            time: 0.0,
            frame: 0,
            resolution: [0, 0],
            _pad2: [0; 4],
        }
    }
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_format: wgpu::TextureFormat,
    pub resources: Resources,
    pub uniforms: Uniforms,
    pub timing_system: TimingSystem,
    pub frame: u32,
    pub last_resolution: [u32; 2],
}

impl Renderer {
    #[cfg(target_arch = "wasm32")]
    pub async fn new(canvas: &web_sys::HtmlCanvasElement) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("Failed to create surface: {:?}", e))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|e| format!("Failed to find adapter: {:?}", e))?;

        let mut required_features = wgpu::Features::empty();
        let supported_features = adapter.features();
        if supported_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Queues Device"),
                required_features,
                required_limits: wgpu::Limits {
                    // Needed to bind the 256 MiB storage buffer used by the queue stress test.
                    max_storage_buffer_binding_size: 256 * 1024 * 1024,
                    ..wgpu::Limits::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("Failed to create device: {:?}", e))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: canvas.width(),
            height: canvas.height(),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        let uniforms = Uniforms {
            time: 0.0,
            frame: 0,
            resolution: [canvas.width(), canvas.height()],
            _pad2: [0; 4],
        };

        let resources = Resources::new(device.clone(), queue.clone());
        let timing_system = TimingSystem::new(device.clone(), queue.clone(), 64);

        Ok(Self {
            device,
            queue,
            surface,
            surface_format: format,
            resources,
            uniforms,
            timing_system,
            frame: 0,
            last_resolution: [0, 0],
        })
    }

    /// Stub for non-wasm32 targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new(_canvas: &web_sys::HtmlCanvasElement) -> Result<Self, String> {
        Err("Renderer is only available on wasm32 targets".to_string())
    }

    pub async fn compile_shaders_async(
        device: &wgpu::Device,
        shaders: &HashMap<String, String>,
    ) -> Result<Vec<CompiledShader>, String> {
        compile_shaders_default(device, shaders)
            .await
            .map_err(|errors| {
                errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
    }

    pub fn install_compiled_shaders(&mut self, compiled: Vec<CompiledShader>) {
        for shader in compiled {
            self.resources
                .add_shader_module(&shader.name, shader.module, Some(shader.source_map));
        }
    }

    pub fn initialize_buffers(&mut self) {
        let width = self.uniforms.resolution[0].max(1);
        let height = self.uniforms.resolution[1].max(1);

        // Uniform buffer
        self.resources.add_buffer(
            "uniforms",
            &[self.uniforms],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        self.resources.add_empty_buffer(
            "queue_buffer",
            256u64 * 1024u64 * 1024u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let cost_buffer_size = (width as u64) * (height as u64) * 4;
        self.resources.add_empty_buffer(
            "cost_buffer",
            cost_buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let output_buffer_size = (width as u64) * (height as u64) * 16; // vec4f = 16 bytes
        self.resources.add_empty_buffer(
            "output_buffer",
            output_buffer_size,
            wgpu::BufferUsages::STORAGE,
        );

        self.resources.add_empty_buffer(
            "active_buffer",
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        self.resources.add_empty_buffer(
            "seeds_buffer",
            1024 * 128, 
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        self.last_resolution = [width, height];

        log::info!("✓ Initialized buffers ({}x{})", width, height);
    }

    fn resize_if_needed(&mut self, width: u32, height: u32) {
        if self.last_resolution == [width, height] || width == 0 || height == 0 {
            return;
        }

        log::info!("Rebuilding for new resolution {}x{}", width, height);

        // Reconfigure surface to match new resolution (same as Starling)
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 1,
            },
        );

        // Resize buffers
        let cost_buffer_size = (width as u64) * (height as u64) * 4;
        self.resources.add_empty_buffer(
            "cost_buffer",
            cost_buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let output_buffer_size = (width as u64) * (height as u64) * 16;
        self.resources.add_empty_buffer(
            "output_buffer",
            output_buffer_size,
            wgpu::BufferUsages::STORAGE,
        );

        self.last_resolution = [width, height];

        if let Err(e) = self.build_pipelines() {
            log::error!("Failed to rebuild pipelines after resize: {}", e);
        }
    }

    pub fn build_pipelines(&mut self) -> Result<(), String> {
        const COMPUTE: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;
        const VERTEX_FRAGMENT: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX_FRAGMENT;

        self.resources.add_compute_pipeline_with_groups(
            "flood_fill_reset",
            "flood_fill_reset",
            "main",
            &[&[
                buffer_entry(COMPUTE, UNIFORM, "uniforms"),
                buffer_entry(COMPUTE, READ_WRITE, "queue_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "cost_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "active_buffer"),
            ]],
        )?;

        self.resources.add_compute_pipeline_with_groups(
            "flood_fill_init",
            "flood_fill_init",
            "main",
            &[&[
                buffer_entry(COMPUTE, UNIFORM, "uniforms"),
                buffer_entry(COMPUTE, READ_WRITE, "queue_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "cost_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "active_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "seeds_buffer"),
            ]],
        )?;

        self.resources.add_compute_pipeline_with_groups(
            "flood_fill_propagate",
            "flood_fill_propagate",
            "main",
            &[&[
                buffer_entry(COMPUTE, UNIFORM, "uniforms"),
                buffer_entry(COMPUTE, READ_WRITE, "queue_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "cost_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "active_buffer"),
            ]],
        )?;

        self.resources.add_compute_pipeline_with_groups(
            "fullscreen",
            "fullscreen",
            "main",
            &[&[
                buffer_entry(COMPUTE, UNIFORM, "uniforms"),
                buffer_entry(COMPUTE, READ, "cost_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "output_buffer"),
            ]],
        )?;

        self.resources.add_compute_pipeline_with_groups(
            "draw_seeds",
            "draw_seeds",
            "main",
            &[&[
                buffer_entry(COMPUTE, UNIFORM, "uniforms"),
                buffer_entry(COMPUTE, READ, "seeds_buffer"),
                buffer_entry(COMPUTE, READ_WRITE, "output_buffer"),
            ]],
        )?;

        // Present pipeline - just reads from output_buffer and applies gamma
        self.resources.add_render_pipeline_with_groups(
            "present_pipeline",
            "present",
            "vs_main",
            "fs_main",
            self.surface_format,
            &[&[
                buffer_entry(VERTEX_FRAGMENT, UNIFORM, "uniforms"),
                buffer_entry(VERTEX_FRAGMENT, READ, "output_buffer"),
            ]],
        )?;

        self.frame = 0;

        log::info!("✓ Built pipelines");
        Ok(())
    }

    /// Renders a single frame.
    pub fn render(&mut self, time: f32, width: u32, height: u32) -> Result<(), String> {
        // Update uniforms
        self.uniforms.time = time;
        self.uniforms.frame = self.frame;
        self.uniforms.resolution = [width, height];
        self.resources.update_buffer("uniforms", &[self.uniforms]);

        // Resize surface and buffers if needed (also rebuilds pipelines)
        self.resize_if_needed(width, height);

        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| format!("{:?}", e))?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.resources
            .dispatch_compute_with_pipeline(
                "flood_fill_reset",
                &mut encoder,
                &mut self.timing_system,
                1024,
                1,
                1,
            )
            .unwrap();

        self.resources
        .dispatch_compute_with_pipeline(
            "flood_fill_init",
            &mut encoder,
            &mut self.timing_system,
            1,
            1,
            1,
        )
        .unwrap();

        self.resources
        .dispatch_compute_with_pipeline(
            "flood_fill_propagate",
            &mut encoder,
            &mut self.timing_system,
            1024,
            1,
            1,
        )
        .unwrap();

        self.resources
            .dispatch_compute_with_pipeline(
                "fullscreen",
                &mut encoder,
                &mut self.timing_system,
                u32::div_ceil(width, 8),
                u32::div_ceil(height, 8),
                1,
            )
            .unwrap();

        self.resources
            .dispatch_compute_with_pipeline(
                "draw_seeds",
                &mut encoder,
                &mut self.timing_system,
                1,
                1,
                1,
            )
            .unwrap();

        self.timing_system.resolve_queries(&mut encoder);

        self.resources.dispatch_render_with_pipeline(
            "present_pipeline",
            &mut encoder,
            &view,
            |pass| {
                pass.draw(0..3, 0..1); // Fullscreen triangle
            },
        )?;

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        self.timing_system.finalize_frame();

        self.frame = self.frame.wrapping_add(1);

        Ok(())
    }

    pub fn get_timing_results(&self) -> Vec<wgpu_utils::TimingResult> {
        self.timing_system.results()
    }
}
