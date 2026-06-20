//! GPU known-answer harness + a dump of each family's declarative assembly and
//! its fused form. Every kernel runs on the GPU and is checked against a
//! closed-form expected result from structural inputs (identity → passthrough,
//! ones → sum, rope@pos0 → identity). No CPU reimplementation of any kernel.
//!
//! `cargo run --bin verify --features verify`.

use chat_wgpu::arch::{self, Block};
use chat_wgpu::context::GpuContext;
use chat_wgpu::families::Family;
use chat_wgpu::kernels::{activation, matmul, norm, rope};

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

    // --- kernel building blocks (GPU known-answer) ---
    fails += matmul_ones(&ctx).await;
    fails += matmul_identity(&ctx).await;
    fails += rmsnorm_plain(&ctx).await;
    fails += rmsnorm_unit(&ctx).await;
    fails += silu(&ctx).await;
    fails += swiglu(&ctx).await;
    fails += geglu(&ctx).await;
    fails += rope_pos0(&ctx).await;

    // --- family assembly + fusion (data) ---
    println!();
    for arch_name in ["qwen3", "qwen2", "llama", "mistral", "gemma2"] {
        report_family(arch_name);
    }

    if fails == 0 {
        println!("\nall kernels verified ✅");
    } else {
        eprintln!("\n{fails} kernel(s) FAILED ❌");
        std::process::exit(1);
    }
}

fn report_family(arch_name: &str) {
    let fam = Family::from_arch(arch_name);
    let fused = arch::fuse(&fam.layer);
    let raw = arch::dispatch_count(&fam.layer);
    let fz = arch::dispatch_count(&fused);
    println!(
        "{arch_name:>8} -> {:<6} layer: {raw} blocks -> {fz} after fusion  (window={:?} softcap={:?})",
        fam.name, fam.params.sliding_window, fam.params.final_logit_softcap
    );
    // Show what fused into what for the first sublayer.
    let preview: Vec<&str> = fused.iter().take(6).map(block_tag).collect();
    println!("           fused head: {}", preview.join(" · "));
}

fn block_tag(b: &Block) -> &'static str {
    use Block::*;
    match b {
        ResidualSave => "save",
        ResidualAdd => "add",
        RmsNorm => "rmsnorm",
        RmsNormUnit => "rmsnorm_unit",
        Linear(_) => "linear",
        QkNorm => "qk_norm",
        Rope => "rope",
        Attention(_) => "attn",
        SwiGlu => "swiglu",
        GeGlu => "geglu",
        FusedNormLinear { .. } => "norm+linear",
        FusedGatedMlp { .. } => "gate+up+act",
    }
}

// --- kernel checks ---

async fn matmul_ones(ctx: &GpuContext) -> u32 {
    let (m, k, n) = (8usize, 1024usize, 32usize);
    let a = ctx.storage(&vec![1.0f32; m * k]);
    let b = ctx.storage(&vec![1.0f32; k * n]);
    let c = matmul::matmul(ctx, &a, &b, m, k, n);
    report("matmul ones", &ctx.read(&c, m * n).await, &vec![k as f32; m * n])
}

async fn matmul_identity(ctx: &GpuContext) -> u32 {
    let n = 64usize;
    let mut id = vec![0f32; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    let bvals: Vec<f32> = (0..n * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let a = ctx.storage(&id);
    let b = ctx.storage(&bvals);
    let c = matmul::matmul(ctx, &a, &b, n, n, n);
    report("matmul identity", &ctx.read(&c, n * n).await, &bvals)
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
    // gain = 1 + w = 2
    report("norm::rmsnorm_unit", &ctx.read(&y, rows * dim).await, &vec![2.0 / (1.0 + eps).sqrt(); rows * dim])
}

async fn silu(ctx: &GpuContext) -> u32 {
    let n = 4096usize;
    let v = 1.5f32;
    let x = ctx.storage(&vec![v; n]);
    let y = activation::silu(ctx, &x, n);
    report("activation::silu", &ctx.read(&y, n).await, &vec![v / (1.0 + (-v).exp()); n])
}

async fn swiglu(ctx: &GpuContext) -> u32 {
    let n = 4096usize;
    let (g, u) = (1.5f32, -0.7f32);
    let gate = ctx.storage(&vec![g; n]);
    let up = ctx.storage(&vec![u; n]);
    let out = activation::swiglu(ctx, &gate, &up, n);
    let silu = g / (1.0 + (-g).exp());
    report("activation::swiglu", &ctx.read(&out, n).await, &vec![silu * u; n])
}

async fn geglu(ctx: &GpuContext) -> u32 {
    let n = 4096usize;
    let (g, u) = (1.5f32, -0.7f32);
    let gate = ctx.storage(&vec![g; n]);
    let up = ctx.storage(&vec![u; n]);
    let out = activation::geglu(ctx, &gate, &up, n);
    let c = 0.797_884_56f32;
    let gelu = 0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh());
    report("activation::geglu", &ctx.read(&out, n).await, &vec![gelu * u; n])
}

async fn rope_pos0(ctx: &GpuContext) -> u32 {
    let (rows, head_dim) = (16usize, 128usize);
    let x: Vec<f32> = (0..rows * head_dim).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
    let xb = ctx.storage(&x);
    let y = rope::rope(ctx, &xb, rows, head_dim, 0, 10_000.0);
    report("rope pos0=identity", &ctx.read(&y, rows * head_dim).await, &x)
}

fn report(name: &str, got: &[f32], expect: &[f32]) -> u32 {
    let max_err = got.iter().zip(expect).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let ok = max_err < 1e-3;
    println!("{name}: max_err={max_err:e}  {}", if ok { "ok" } else { "FAIL" });
    u32::from(!ok)
}
