//! GPU known-answer harness for the kernel building blocks, plus a Qwen3
//! forward run on random weights that exercises the `Model` trait and the `Hook`
//! intermediate-tap. Every kernel is checked against a closed-form expected
//! result; no CPU reimplementation of any kernel.
//!
//! `cargo run --bin verify --features verify`.

use std::collections::HashMap;

use chat_wgpu::context::GpuContext;
use chat_wgpu::kernels::{activation, attention, matmul, norm, rope};
use chat_wgpu::model::qwen3::Qwen3;
use chat_wgpu::model::{self, Hook, Model, Weights};

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let ctx = match GpuContext::new().await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU: {e}");
            std::process::exit(1);
        }
    };
    println!("backend: {}\n", ctx.backend);
    let mut fails = 0;

    fails += matmul_ones(&ctx).await;
    fails += rmsnorm_plain(&ctx).await;
    fails += rmsnorm_unit(&ctx).await;
    fails += swiglu(&ctx).await;
    fails += geglu(&ctx).await;
    fails += rope_pos0(&ctx).await;
    fails += attention_single_key(&ctx).await;
    fails += ewise_add(&ctx).await;

    println!();
    qwen3_forward(&ctx).await;

    if fails == 0 {
        println!("\nall kernels verified ✅");
    } else {
        eprintln!("\n{fails} kernel(s) FAILED ❌");
        std::process::exit(1);
    }
}

// ── kernel known-answer checks ──

async fn matmul_ones(ctx: &GpuContext) -> u32 {
    let (m, k, n) = (8usize, 1024usize, 32usize);
    let a = ctx.storage(&vec![1.0f32; m * k]);
    let b = ctx.storage(&vec![1.0f32; k * n]);
    let c = matmul::matmul(ctx, &a, &b, m, k, n);
    report("matmul ones", &ctx.read(&c, m * n).await, &vec![k as f32; m * n])
}

async fn rmsnorm_plain(ctx: &GpuContext) -> u32 {
    let (rows, dim, eps) = (4usize, 1024usize, 1e-6f32);
    let x = ctx.storage(&vec![1.0f32; rows * dim]);
    let w = ctx.storage(&vec![1.0f32; dim]);
    let y = norm::rmsnorm(ctx, &x, &w, rows, dim, eps);
    report("norm::rmsnorm", &ctx.read(&y, rows * dim).await, &vec![1.0 / (1.0 + eps).sqrt(); rows * dim])
}

async fn rmsnorm_unit(ctx: &GpuContext) -> u32 {
    let (rows, dim, eps) = (4usize, 1024usize, 1e-6f32);
    let x = ctx.storage(&vec![1.0f32; rows * dim]);
    let w = ctx.storage(&vec![1.0f32; dim]);
    let y = norm::rmsnorm_unit(ctx, &x, &w, rows, dim, eps);
    report("norm::rmsnorm_unit", &ctx.read(&y, rows * dim).await, &vec![2.0 / (1.0 + eps).sqrt(); rows * dim])
}

async fn swiglu(ctx: &GpuContext) -> u32 {
    let n = 4096usize;
    let (g, u) = (1.5f32, -0.7f32);
    let out = activation::swiglu(ctx, &ctx.storage(&vec![g; n]), &ctx.storage(&vec![u; n]), n);
    report("activation::swiglu", &ctx.read(&out, n).await, &vec![g / (1.0 + (-g).exp()) * u; n])
}

async fn geglu(ctx: &GpuContext) -> u32 {
    let n = 4096usize;
    let (g, u) = (1.5f32, -0.7f32);
    let out = activation::geglu(ctx, &ctx.storage(&vec![g; n]), &ctx.storage(&vec![u; n]), n);
    let c = 0.797_884_56f32;
    let gelu = 0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh());
    report("activation::geglu", &ctx.read(&out, n).await, &vec![gelu * u; n])
}

async fn rope_pos0(ctx: &GpuContext) -> u32 {
    let (rows, head_dim) = (16usize, 128usize);
    let x: Vec<f32> = (0..rows * head_dim).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
    let y = rope::rope(ctx, &ctx.storage(&x), rows, head_dim, 0, 10_000.0);
    report("rope pos0=identity", &ctx.read(&y, rows * head_dim).await, &x)
}

/// One key (seq=1) → softmax over a single score = 1 → output equals V.
async fn attention_single_key(ctx: &GpuContext) -> u32 {
    let (n_heads, n_kv, head_dim) = (4usize, 2usize, 16usize);
    let q: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.01).collect();
    let v: Vec<f32> = (0..n_kv * head_dim).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
    let k = vec![0.5f32; n_kv * head_dim];
    let out = attention::attention(
        ctx, &ctx.storage(&q), &ctx.storage(&k), &ctx.storage(&v), n_heads, n_kv, 1, head_dim,
    );
    // out[head] = V[kv(head)]; kv = head / (n_heads/n_kv).
    let mut expect = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv = h / (n_heads / n_kv);
        for d in 0..head_dim {
            expect[h * head_dim + d] = v[kv * head_dim + d];
        }
    }
    report("attention seq=1=V", &ctx.read(&out, n_heads * head_dim).await, &expect)
}

async fn ewise_add(ctx: &GpuContext) -> u32 {
    let n = 1000usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.5).collect();
    let c = attention::add(ctx, &ctx.storage(&a), &ctx.storage(&b), n);
    let expect: Vec<f32> = (0..n).map(|i| i as f32 - i as f32 * 0.5).collect();
    report("ewise::add", &ctx.read(&c, n).await, &expect)
}

// ── Qwen3 forward on random weights + an intermediate tap ──

async fn qwen3_forward(ctx: &GpuContext) {
    let mut weights = RandomWeights {
        dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        hidden: 128,
        vocab: 256,
    };
    let model = Qwen3::load(ctx, &mut weights).expect("load");

    // Greedy generation over the full pipeline: embed → layers + KV cache →
    // lm-head → argmax → feed back. Random weights, so the tokens are gibberish;
    // the point is the decode loop + cache run end-to-end on GPU.
    let prompt = [1u32, 5, 9, 2];
    let mut cap = Capture::default();
    let out = model::generate(ctx, &model, &prompt, 8, &mut cap).await;
    println!("qwen3 generate (random weights): prompt {prompt:?} -> {out:?}");

    // Pull a mid-layer intermediate back out — the meta-ML use case.
    if let Some((buf, len)) = cap.get("post_attn", 0) {
        let mid = ctx.read(&buf, len).await;
        println!("tapped post_attn @layer0[0..4] = {:?}", &mid[..4]);
    }
    println!("hook captured {} distinct tap points", cap.taps.len());
}

/// A `Hook` that stashes every tapped buffer (handles are cheap clones).
#[derive(Default)]
struct Capture {
    taps: HashMap<(String, usize), (wgpu::Buffer, usize)>,
}
impl Capture {
    fn get(&self, name: &str, layer: usize) -> Option<(wgpu::Buffer, usize)> {
        self.taps.get(&(name.to_string(), layer)).cloned()
    }
}
impl Hook for Capture {
    fn tap(&mut self, name: &str, layer: usize, buf: &wgpu::Buffer, len: usize) {
        self.taps.insert((name.to_string(), layer), (buf.clone(), len));
    }
}

/// Random weight source: deterministic pseudo-random buffers + fixed metadata.
struct RandomWeights {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    vocab: usize,
}
impl Weights for RandomWeights {
    fn meta_u32(&self, key: &str) -> u32 {
        (match key {
            k if k.ends_with("block_count") => self.n_layers,
            k if k.ends_with("embedding_length") => self.dim,
            k if k.ends_with("head_count_kv") => self.n_kv_heads,
            k if k.ends_with("head_count") => self.n_heads,
            k if k.ends_with("key_length") => self.head_dim,
            k if k.ends_with("feed_forward_length") => self.hidden,
            k if k.ends_with("vocab_size") => self.vocab,
            _ => 0,
        }) as u32
    }
    fn meta_f32(&self, key: &str) -> f32 {
        match key {
            k if k.ends_with("rms_epsilon") => 1e-6,
            k if k.ends_with("freq_base") => 1_000_000.0,
            _ => 0.0,
        }
    }
    fn has(&self, _name: &str) -> bool {
        false // no biases in the random model
    }
    fn matrix(&mut self, ctx: &GpuContext, name: &str, in_f: usize, out_f: usize) -> wgpu::Buffer {
        ctx.storage(&fill(seed(name), in_f * out_f))
    }
    fn vector(&mut self, ctx: &GpuContext, name: &str, len: usize) -> wgpu::Buffer {
        ctx.storage(&fill(seed(name), len))
    }
}

fn seed(name: &str) -> u64 {
    name.bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211))
}

/// Deterministic small values in ~[-0.1, 0.1] from an LCG.
fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2
        })
        .collect()
}

fn report(name: &str, got: &[f32], expect: &[f32]) -> u32 {
    let max_err = got.iter().zip(expect).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let ok = max_err < 1e-3;
    println!("{name}: max_err={max_err:e}  {}", if ok { "ok" } else { "FAIL" });
    u32::from(!ok)
}
