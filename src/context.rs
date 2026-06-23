//! GPU context: device/queue, a pipeline cache, and minimal buffer/dispatch
//! helpers. Backend-agnostic wgpu, so the same code runs natively (Metal here)
//! and, later, on WebGPU in the browser.
//!
//! Convention: every kernel binds its storage buffers at sequential `@binding`
//! indices starting at 0, followed by a single uniform at the last index. The
//! `run` helper builds the bind group from the buffers in the order given.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::profile::Profiler;

/// A compute pipeline plus its kernel name. The name becomes the GPU
/// compute-pass label (visible in Xcode / Metal captures) and the key the
/// timestamp profiler aggregates by. Built by [`GpuContext::pipeline`].
#[derive(Clone)]
pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub name: &'static str,
}

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
    /// Uniform buffers are immutable and their contents (matmul dims, …) repeat
    /// every token — cache by bytes instead of reallocating ~hundreds/token.
    uniforms: RefCell<HashMap<Vec<u8>, wgpu::Buffer>>,
    /// Deterministic per-token scratch arena: `empty` hands out arena[cursor++],
    /// `reset_frame` rewinds the cursor. Because the decode forward issues the
    /// same op sequence every token, op K always gets the same buffer — which is
    /// what lets the bind-group cache stay valid.
    arena: RefCell<Vec<(usize, wgpu::Buffer)>>,
    cursor: Cell<usize>,
    /// Bind groups cached by op index (the count of `run` calls since
    /// `reset_frame`). Built once, reused every token — the big win, since
    /// `create_bind_group` is wgpu's ~0.3 ms/op cost.
    op_index: Cell<usize>,
    bg_cache: RefCell<Vec<Option<wgpu::BindGroup>>>,
    /// GPU timestamp profiler. `None` unless `begin_profile` was called this
    /// frame (and the adapter supports TIMESTAMP_QUERY). See `src/profile.rs`.
    profiler: RefCell<Option<Profiler>>,
    timestamps_supported: bool,
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
        // Subgroup ops (subgroupAdd) for fast GEMV reductions, and timestamp
        // queries for the GPU profiler — request whichever the adapter exposes
        // (Metal/Vulkan/DX12 support both).
        let wanted = wgpu::Features::SUBGROUP | wgpu::Features::TIMESTAMP_QUERY;
        let features = adapter.features() & wanted;
        let timestamps_supported = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
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
            uniforms: RefCell::new(HashMap::new()),
            arena: RefCell::new(Vec::new()),
            cursor: Cell::new(0),
            op_index: Cell::new(0),
            bg_cache: RefCell::new(Vec::new()),
            profiler: RefCell::new(None),
            timestamps_supported,
        })
    }

    /// Start capturing per-kernel GPU timestamps for the next frame. Pair with
    /// [`report_profile`](Self::report_profile) after the frame's readback.
    /// No-op (with a warning) if the adapter lacks TIMESTAMP_QUERY.
    pub fn begin_profile(&self) {
        if !self.timestamps_supported {
            eprintln!("[profile] TIMESTAMP_QUERY unsupported on this adapter — skipping");
            return;
        }
        *self.profiler.borrow_mut() = Some(Profiler::new(&self.device, &self.queue, 2048));
    }

    /// Finish a profile started with [`begin_profile`](Self::begin_profile):
    /// map the timestamps and aggregate per kernel. Returns
    /// `(kernel, passes, total_ms)` sorted by total. Call after `read`.
    pub async fn report_profile(&self) -> Vec<(&'static str, u32, f64)> {
        let prof = self.profiler.borrow_mut().take();
        match prof {
            Some(p) => p.report(&self.device).await,
            None => vec![],
        }
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
        if let Some(mut enc) = self.encoder.borrow_mut().take() {
            // Resolve profiler timestamps into the same submit, if profiling.
            if let Some(p) = self.profiler.borrow().as_ref() {
                p.resolve(&mut enc);
            }
            self.queue.submit([enc.finish()]);
            self.retained_bg.borrow_mut().clear();
            self.retained_buf.borrow_mut().clear();
        }
    }

    /// Get-or-build a compute pipeline for `(name, wgsl, entry)`, cached by name.
    /// Returns a [`Kernel`] so the name rides along to the pass label + profiler.
    pub fn pipeline(&self, name: &'static str, wgsl: &str, entry: &str) -> Kernel {
        if let Some(p) = self.pipelines.borrow().get(name) {
            return Kernel { pipeline: p.clone(), name };
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
        Kernel { pipeline, name }
    }

    pub fn storage(&self, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Upload packed u32 data (quantized weights).
    pub fn storage_u32(&self, data: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    fn alloc(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// A scratch buffer of `len` f32s from the per-token arena (slot `cursor`).
    /// The same slot returns the same buffer every token (deterministic op
    /// sequence), which keeps cached bind groups valid.
    pub fn empty(&self, len: usize) -> wgpu::Buffer {
        let c = self.cursor.get();
        self.cursor.set(c + 1);
        let mut arena = self.arena.borrow_mut();
        if let Some((sz, buf)) = arena.get(c) {
            if *sz == len {
                return buf.clone();
            }
            // Size changed for this slot — reallocate (and any cached bind group
            // for an op using it rebuilds, since contents differ on cache miss).
            let b = self.alloc(len);
            arena[c] = (len, b.clone());
            return b;
        }
        let b = self.alloc(len);
        arena.push((len, b.clone()));
        b
    }

    /// A buffer that outlives a token (KV cache, weights) — never recycled.
    pub fn empty_persistent(&self, len: usize) -> wgpu::Buffer {
        self.alloc(len)
    }

    /// Rewind the arena + op-index for a new token, keeping the bind-group cache
    /// (the op sequence repeats). Call only when the GPU is idle for the
    /// previous token (i.e. right after `read`).
    pub fn reset_frame(&self) {
        self.cursor.set(0);
        self.op_index.set(0);
    }

    /// Drop the arena and bind-group cache entirely — call when starting a new
    /// program whose op sequence differs from whatever ran before (e.g. at the
    /// start of generation, after unrelated kernels). Must be GPU-idle.
    pub fn clear_cache(&self) {
        self.cursor.set(0);
        self.op_index.set(0);
        self.arena.borrow_mut().clear();
        self.bg_cache.borrow_mut().clear();
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
        if let Some(b) = self.uniforms.borrow().get(bytes) {
            return b.clone();
        }
        let b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        self.uniforms.borrow_mut().insert(bytes.to_vec(), b.clone());
        b
    }

    fn make_bind_group(&self, kernel: &Kernel, binds: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel.name),
            layout: &kernel.pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    fn dispatch(&self, kernel: &Kernel, bind_group: &wgpu::BindGroup, wg: (u32, u32, u32)) {
        let pipeline = kernel.pipeline.clone();
        let bind_group = bind_group.clone();
        let name = kernel.name;
        // Reserve a begin/end timestamp pair if a profile is active. The borrow
        // of `profiler` (and thus the QuerySet) is held across `record`, which
        // runs the closure synchronously, so the &QuerySet stays valid.
        let prof = self.profiler.borrow();
        let pair = prof
            .as_ref()
            .and_then(|p| p.reserve(name).map(|(b, e)| (p.query_set(), b, e)));
        self.record(move |enc| {
            let timestamp_writes = pair.map(|(qs, b, e)| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
        });
        drop(prof);
    }

    /// Dispatch with a **cached** bind group (keyed by op index within the
    /// frame). The bind group is built once and reused every token — this is
    /// what removes wgpu's ~0.3 ms/op `create_bind_group` cost. Safe only when
    /// the buffers bound at this op index are the same every token (arena +
    /// fixed op sequence), and the uniform content doesn't vary per token.
    pub fn run(&self, kernel: &Kernel, binds: &[&wgpu::Buffer], workgroups: (u32, u32, u32)) {
        let idx = self.op_index.get();
        self.op_index.set(idx + 1);
        let mut cache = self.bg_cache.borrow_mut();
        while cache.len() <= idx {
            cache.push(None);
        }
        let bg = cache[idx]
            .get_or_insert_with(|| self.make_bind_group(kernel, binds))
            .clone();
        drop(cache);
        self.dispatch(kernel, &bg, workgroups);
    }

    /// Dispatch with a fresh bind group every call — for ops whose bound
    /// uniform varies per token (RoPE position, attention seq length), so the
    /// cache can't apply. Still advances the op index to keep cache alignment.
    pub fn run_uncached(&self, kernel: &Kernel, binds: &[&wgpu::Buffer], workgroups: (u32, u32, u32)) {
        self.op_index.set(self.op_index.get() + 1);
        let bg = self.make_bind_group(kernel, binds);
        self.dispatch(kernel, &bg, workgroups);
        self.retained_bg.borrow_mut().push(bg);
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
