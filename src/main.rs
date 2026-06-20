//! GPU known-answer harness. Every kernel runs on the GPU and is checked
//! against a closed-form expected result derived from structural inputs
//! (identity → passthrough, ones → sum, rope@pos0 → identity, …). No CPU
//! reimplementation of any kernel — correctness is established on-device.
//!
//! `cargo run --bin verify --features verify`.

use chat_wgpu::context::GpuContext;
use chat_wgpu::kernels::{self, Activation, NormKind};

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
    println!("backend: {}", ctx.backend);
    let mut fails = 0;

    fails += matmul_ones(&ctx).await;
    fails += matmul_identity(&ctx).await;
    fails += rmsnorm_ones(&ctx).await;
    fails += rmsnorm_gemma(&ctx).await;
    fails += glu_swiglu(&ctx).await;
    fails += glu_geglu(&ctx).await;
    fails += rope_pos0_identity(&ctx).await;
    fails += rope_quarter_turn(&ctx).await;

    // Family resolution sanity: each arch maps to the right kernel variants.
    use chat_wgpu::families::FamilySpec;
    for arch in ["qwen3", "qwen2", "llama", "mistral", "gemma2"] {
        let s = FamilySpec::from_arch(arch);
        println!(
            "family {arch:>8} -> {:<6} norm={:?} act={:?} qk_norm={} qkv_bias={}",
            s.name, s.norm, s.activation, s.use_qk_norm, s.attn_qkv_bias
        );
    }

    if fails == 0 {
        println!("\nall kernels verified ✅");
    } else {
        eprintln!("\n{fails} kernel(s) FAILED ❌");
        std::process::exit(1);
    }
}

/// `ones(m,k) · ones(k,n)` → every element equals `k`.
async fn matmul_ones(ctx: &GpuContext) -> u32 {
    let (m, k, n) = (8usize, 1024usize, 32usize);
    let a = ctx.storage(&vec![1.0f32; m * k]);
    let b = ctx.storage(&vec![1.0f32; k * n]);
    let c = kernels::matmul(ctx, &a, &b, m, k, n);
    let got = ctx.read(&c, m * n).await;
    report("matmul ones", &got, &vec![k as f32; m * n])
}

/// `I(n) · B` → `B` (passthrough).
async fn matmul_identity(ctx: &GpuContext) -> u32 {
    let n = 64usize;
    let mut id = vec![0f32; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    let bvals: Vec<f32> = (0..n * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let a = ctx.storage(&id);
    let b = ctx.storage(&bvals);
    let c = kernels::matmul(ctx, &a, &b, n, n, n);
    let got = ctx.read(&c, n * n).await;
    report("matmul identity", &got, &bvals)
}

/// Plain `rmsnorm(ones, weight=ones)` → every element equals `1/sqrt(1+eps)`.
async fn rmsnorm_ones(ctx: &GpuContext) -> u32 {
    let (rows, dim, eps) = (4usize, 1024usize, 1e-6f32);
    let x = ctx.storage(&vec![1.0f32; rows * dim]);
    let w = ctx.storage(&vec![1.0f32; dim]);
    let y = kernels::rmsnorm(ctx, &x, &w, rows, dim, eps, NormKind::Plain);
    let got = ctx.read(&y, rows * dim).await;
    let expect = vec![1.0 / (1.0 + eps).sqrt(); rows * dim];
    report("rmsnorm plain ones", &got, &expect)
}

/// Gemma `rmsnorm(ones, weight=ones)` → gain is `1+w = 2`, so `2/sqrt(1+eps)`.
async fn rmsnorm_gemma(ctx: &GpuContext) -> u32 {
    let (rows, dim, eps) = (4usize, 1024usize, 1e-6f32);
    let x = ctx.storage(&vec![1.0f32; rows * dim]);
    let w = ctx.storage(&vec![1.0f32; dim]);
    let y = kernels::rmsnorm(ctx, &x, &w, rows, dim, eps, NormKind::UnitShift);
    let got = ctx.read(&y, rows * dim).await;
    let expect = vec![2.0 / (1.0 + eps).sqrt(); rows * dim];
    report("rmsnorm gemma ones", &got, &expect)
}

/// SwiGLU at constant inputs: `out = silu(g)*u` evaluated at the definition.
async fn glu_swiglu(ctx: &GpuContext) -> u32 {
    let n = 4096usize;
    let (g, u) = (1.5f32, -0.7f32);
    let gate = ctx.storage(&vec![g; n]);
    let up = ctx.storage(&vec![u; n]);
    let out = kernels::glu(ctx, &gate, &up, n, Activation::SwiGlu);
    let got = ctx.read(&out, n).await;
    let silu = g / (1.0 + (-g).exp());
    report("glu swiglu const", &got, &vec![silu * u; n])
}

/// GeGLU at constant inputs: `out = gelu_tanh(g)*u` evaluated at the definition.
async fn glu_geglu(ctx: &GpuContext) -> u32 {
    let n = 4096usize;
    let (g, u) = (1.5f32, -0.7f32);
    let gate = ctx.storage(&vec![g; n]);
    let up = ctx.storage(&vec![u; n]);
    let out = kernels::glu(ctx, &gate, &up, n, Activation::GeGlu);
    let got = ctx.read(&out, n).await;
    let c = 0.797_884_56f32; // sqrt(2/pi)
    let gelu = 0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh());
    report("glu geglu const", &got, &vec![gelu * u; n])
}

/// RoPE at position 0 → rotation by zero angle → identity.
async fn rope_pos0_identity(ctx: &GpuContext) -> u32 {
    let (rows, head_dim) = (16usize, 128usize);
    let x: Vec<f32> = (0..rows * head_dim).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
    let xb = ctx.storage(&x);
    let y = kernels::rope(ctx, &xb, rows, head_dim, 0, 10_000.0);
    let got = ctx.read(&y, rows * head_dim).await;
    report("rope pos0=identity", &got, &x)
}

/// RoPE on head_dim=2: half=1, i=0 → freq = pos·theta^0 = pos = 1 rad. Check the
/// pair rotates by exactly 1 radian: (lo,hi) -> (lo·cos1 - hi·sin1, hi·cos1 + lo·sin1).
async fn rope_quarter_turn(ctx: &GpuContext) -> u32 {
    let rows = 4usize;
    let head_dim = 2usize;
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.0, 2.0];
    let pos = 1usize;
    let (c, s) = (1.0f32.cos(), 1.0f32.sin());
    let xb = ctx.storage(&x);
    let y = kernels::rope(ctx, &xb, rows, head_dim, pos, 10_000.0);
    let got = ctx.read(&y, rows * head_dim).await;
    let mut expect = vec![0f32; rows * head_dim];
    for r in 0..rows {
        let (lo, hi) = (x[r * 2], x[r * 2 + 1]);
        expect[r * 2] = lo * c - hi * s;
        expect[r * 2 + 1] = hi * c + lo * s;
    }
    report("rope angle=1rad", &got, &expect)
}

/// Compare GPU output to the closed-form expected values; print max abs error.
fn report(name: &str, got: &[f32], expect: &[f32]) -> u32 {
    let max_err = got
        .iter()
        .zip(expect)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let ok = max_err < 1e-3;
    println!("{name}: max_err={max_err:e}  {}", if ok { "ok" } else { "FAIL" });
    u32::from(!ok)
}
