//! Reusable weight-bearing components, shared across models. Each wraps the
//! weights it owns and exposes a `forward` over the kernel building blocks.

use super::Weights;
use crate::context::GpuContext;
use crate::kernels::{attention, matmul, norm};

/// Linear projection. Weight stored `[in_f, out_f]` (the matmul B operand);
/// optional bias `[out_f]`.
pub struct Linear {
    w: wgpu::Buffer,
    bias: Option<wgpu::Buffer>,
    in_f: usize,
    out_f: usize,
}

impl Linear {
    pub fn load(ctx: &GpuContext, w: &mut dyn Weights, name: &str, in_f: usize, out_f: usize) -> Self {
        let bias_name = format!("{name}.bias");
        let bias = w.has(&bias_name).then(|| w.vector(ctx, &bias_name, out_f));
        let mat = w.matrix(ctx, &format!("{name}.weight"), in_f, out_f);
        Self { w: mat, bias, in_f, out_f }
    }

    /// `x: [rows, in_f] -> [rows, out_f]`.
    pub fn forward(&self, ctx: &GpuContext, x: &wgpu::Buffer, rows: usize) -> wgpu::Buffer {
        let y = matmul::matmul(ctx, x, &self.w, rows, self.in_f, self.out_f);
        match (&self.bias, rows) {
            // Decode path (rows == 1): bias is a plain elementwise add.
            (Some(b), 1) => attention::add(ctx, &y, b, self.out_f),
            _ => y,
        }
    }

    pub fn out_f(&self) -> usize {
        self.out_f
    }
}

/// RMSNorm gain weight. `unit` picks the `1 + weight` (Gemma) kernel building
/// block; otherwise the plain (Llama / Qwen) one.
pub struct RmsNorm {
    w: wgpu::Buffer,
    dim: usize,
    eps: f32,
    unit: bool,
}

impl RmsNorm {
    pub fn load(
        ctx: &GpuContext,
        w: &mut dyn Weights,
        name: &str,
        dim: usize,
        eps: f32,
        unit: bool,
    ) -> Self {
        let g = w.vector(ctx, &format!("{name}.weight"), dim);
        Self { w: g, dim, eps, unit }
    }

    /// Normalize `rows` rows of length `dim` (e.g. one hidden state, or `n_heads`
    /// head vectors for per-head QK-Norm).
    pub fn forward(&self, ctx: &GpuContext, x: &wgpu::Buffer, rows: usize) -> wgpu::Buffer {
        if self.unit {
            norm::rmsnorm_unit(ctx, x, &self.w, rows, self.dim, self.eps)
        } else {
            norm::rmsnorm(ctx, x, &self.w, rows, self.dim, self.eps)
        }
    }
}
