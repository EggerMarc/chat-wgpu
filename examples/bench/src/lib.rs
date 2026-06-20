//! WebGPU feasibility spike.
//!
//! Runs an f32 matmul (the op that dominates LLM decode) as a WGSL compute
//! shader and compares it to a naive CPU matmul of the same size. The point is
//! a hard number: how much faster is WebGPU than the CPU-wasm path we have
//! today — i.e. is porting the engine to a WebGPU backend worth it?
//!
//! `bench()` is pure async wgpu, so it runs both natively (Metal/Vulkan/DX, via
//! `pollster` in `main.rs`) and in the browser (WebGPU, via the wasm wrapper).

use wgpu::util::DeviceExt;

const WGSL: &str = r#"
struct Dims { m: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= d.m || col >= d.n) { return; }
    var sum = 0.0;
    for (var kk: u32 = 0u; kk < d.k; kk = kk + 1u) {
        sum = sum + A[row * d.k + kk] * B[kk * d.n + col];
    }
    C[row * d.n + col] = sum;
}
"#;

pub struct BenchResult {
    pub backend: String,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub iters: usize,
    pub gpu_ms: f64,
    pub cpu_ms: f64,
    pub gpu_gflops: f64,
    pub cpu_gflops: f64,
    pub speedup: f64,
    pub max_abs_err: f32,
}

impl BenchResult {
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"backend":"{}","m":{},"k":{},"n":{},"iters":{},"gpu_ms":{:.3},"cpu_ms":{:.3},"gpu_gflops":{:.1},"cpu_gflops":{:.2},"speedup":{:.1},"max_abs_err":{:e}}}"#,
            self.backend,
            self.m,
            self.k,
            self.n,
            self.iters,
            self.gpu_ms,
            self.cpu_ms,
            self.gpu_gflops,
            self.cpu_gflops,
            self.speedup,
            self.max_abs_err,
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
        * 1000.0
}
#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

/// Benchmark an `m x k` · `k x n` matmul on GPU (WGSL) vs naive CPU.
pub async fn bench(m: usize, k: usize, n: usize, iters: usize) -> Result<BenchResult, String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("no WebGPU adapter: {e:?}"))?;
    let backend = format!("{:?}", adapter.get_info().backend);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            // Request the adapter's max so larger buffers fit. NOTE: in the
            // browser, WebGPU caps maxStorageBufferBindingSize (often 128 MB) —
            // a full fp32 lm-head (vocab×dim) won't fit one binding, so a real
            // port must quantize/fp16 or tile it.
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| format!("no device: {e:?}"))?;

    // Deterministic inputs.
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.02).collect();

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A"),
        contents: bytemuck::cast_slice(&a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B"),
        contents: bytemuck::cast_slice(&b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let c_size = (m * n * 4) as u64;
    let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: c_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dims = [m as u32, k as u32, n as u32, 0u32];
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bgl = pipeline.get_bind_group_layout(0);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
        ],
    });

    let wgx = (n as u32).div_ceil(16);
    let wgy = (m as u32).div_ceil(16);
    let dispatch = |label: &str| {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(label),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(wgx, wgy, 1);
        }
        queue.submit([enc.finish()]);
    };

    // Warm up (shader compile + first dispatch), then read back to sync.
    dispatch("warmup");
    let cpu = naive_matmul(&a, &b, m, k, n);
    let gpu_c = read_back(&device, &queue, &c_buf, c_size).await?;
    let max_abs_err = cpu
        .iter()
        .zip(gpu_c.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);

    // Timed GPU: submit `iters` dispatches, then a readback that drains the queue.
    let t0 = now_ms();
    for _ in 0..iters {
        dispatch("timed");
    }
    let _ = read_back(&device, &queue, &c_buf, c_size).await?;
    let gpu_ms = (now_ms() - t0) / iters as f64;

    // Timed CPU: one naive matmul.
    let t1 = now_ms();
    let _ = naive_matmul(&a, &b, m, k, n);
    let cpu_ms = now_ms() - t1;

    let flop = 2.0 * m as f64 * k as f64 * n as f64;
    let gpu_gflops = flop / (gpu_ms / 1000.0) / 1e9;
    let cpu_gflops = flop / (cpu_ms / 1000.0) / 1e9;

    Ok(BenchResult {
        backend,
        m,
        k,
        n,
        iters,
        gpu_ms,
        cpu_ms,
        gpu_gflops,
        cpu_gflops,
        speedup: cpu_ms / gpu_ms,
        max_abs_err,
    })
}

fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Copy the GPU result buffer into a staging buffer and map it (works on native
/// via `poll`, and in the browser via the event loop).
async fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<f32>, String> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.await
        .map_err(|e| format!("map cancelled: {e:?}"))?
        .map_err(|e| format!("map failed: {e:?}"))?;

    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::prelude::*;

    /// Run the matmul benchmark; resolves to a JSON string of timings.
    #[wasm_bindgen]
    pub async fn bench(m: usize, k: usize, n: usize, iters: usize) -> Result<String, JsValue> {
        console_error_panic_hook::set_once();
        super::bench(m, k, n, iters)
            .await
            .map(|r| r.to_json())
            .map_err(|e| JsValue::from_str(&e))
    }
}
