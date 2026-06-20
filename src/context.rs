//! GPU context: device/queue, a pipeline cache, and minimal buffer/dispatch
//! helpers. Backend-agnostic wgpu, so the same code runs natively (Metal here)
//! and, later, on WebGPU in the browser.
//!
//! Convention: every kernel binds its storage buffers at sequential `@binding`
//! indices starting at 0, followed by a single uniform at the last index. The
//! `run` helper builds the bind group from the buffers in the order given.

use std::cell::RefCell;
use std::collections::HashMap;

use wgpu::util::DeviceExt;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub backend: String,
    pipelines: RefCell<HashMap<&'static str, wgpu::ComputePipeline>>,
    /// Ops record into this shared encoder and are submitted in one batch at the
    /// next `read`/`flush`, instead of one `queue.submit` per op. Hundreds of
    /// dispatches per token → one command buffer.
    encoder: RefCell<Option<wgpu::CommandEncoder>>,
    /// Resources referenced by the pending encoder must outlive it until submit
    /// (the caller's buffer handles may drop sooner). Bind groups hold their
    /// buffers; `copy` has no bind group, so its buffers are retained directly.
    retained_bg: RefCell<Vec<wgpu::BindGroup>>,
    retained_buf: RefCell<Vec<wgpu::Buffer>>,
}

impl GpuContext {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no adapter: {e:?}"))?;
        let backend = format!("{:?}", adapter.get_info().backend);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("no device: {e:?}"))?;
        Ok(Self {
            device,
            queue,
            backend,
            pipelines: RefCell::new(HashMap::new()),
            encoder: RefCell::new(None),
            retained_bg: RefCell::new(Vec::new()),
            retained_buf: RefCell::new(Vec::new()),
        })
    }

    /// Run a closure with the shared command encoder, lazily creating it.
    fn record(&self, f: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let mut slot = self.encoder.borrow_mut();
        let enc = slot.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
        });
        f(enc);
    }

    /// Submit any pending recorded ops as one command buffer, then release the
    /// resources that were keeping them alive.
    pub fn flush(&self) {
        if let Some(enc) = self.encoder.borrow_mut().take() {
            self.queue.submit([enc.finish()]);
            self.retained_bg.borrow_mut().clear();
            self.retained_buf.borrow_mut().clear();
        }
    }

    /// Get-or-build a compute pipeline for `(name, wgsl, entry)`, cached by name.
    pub fn pipeline(&self, name: &'static str, wgsl: &str, entry: &str) -> wgpu::ComputePipeline {
        if let Some(p) = self.pipelines.borrow().get(name) {
            return p.clone();
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        self.pipelines.borrow_mut().insert(name, pipeline.clone());
        pipeline
    }

    pub fn storage(&self, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    pub fn empty(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// GPU→GPU copy of `len` f32s from `src[src_off..]` into `dst[dst_off..]`.
    /// Used to write a new token's K/V into the preallocated KV cache, and to
    /// gather an embedding row.
    pub fn copy(&self, src: &wgpu::Buffer, src_off: usize, dst: &wgpu::Buffer, dst_off: usize, len: usize) {
        self.record(|enc| {
            enc.copy_buffer_to_buffer(
                src,
                (src_off * 4) as u64,
                dst,
                (dst_off * 4) as u64,
                (len * 4) as u64,
            );
        });
        // Keep both buffers alive until the batch submits.
        let mut bufs = self.retained_buf.borrow_mut();
        bufs.push(src.clone());
        bufs.push(dst.clone());
    }

    pub fn uniform(&self, bytes: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Dispatch a pipeline over `binds` (sequential bindings) for `workgroups`.
    pub fn run(&self, pipeline: &wgpu::ComputePipeline, binds: &[&wgpu::Buffer], workgroups: (u32, u32, u32)) {
        let entries: Vec<wgpu::BindGroupEntry> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        // Record into the shared encoder; submitted in a batch at the next read.
        // Keep the bind group (and thus its buffers) alive until that submit.
        let pipeline = pipeline.clone();
        self.record(|enc| {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        });
        self.retained_bg.borrow_mut().push(bind_group);
    }

    /// Read a storage buffer of `len` f32s back to the host. Flushes any pending
    /// recorded ops first, so the readback sees their results.
    pub async fn read(&self, src: &wgpu::Buffer, len: usize) -> Vec<f32> {
        self.flush();
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.await.unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }
}
