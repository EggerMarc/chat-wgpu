use super::Weights;
use crate::context::GpuContext;
use crate::kernels::{attention, matmul, norm, q4};

pub struct Linear {
    w: wgpu::Buffer,
    bias: Option<wgpu::Buffer>,
    in_f: usize,
    out_f: usize,
}

impl Linear {
    pub fn load(
        ctx: &GpuContext,
        w: &mut dyn Weights,
        name: &str,
        in_f: usize,
        out_f: usize,
    ) -> Self {
        let bias_name = format!("{name}.bias");
        let bias = w.has(&bias_name).then(|| w.vector(ctx, &bias_name, out_f));
        let mat = w.matrix(ctx, &format!("{name}.weight"), in_f, out_f);
        Self {
            w: mat,
            bias,
            in_f,
            out_f,
        }
    }

    pub fn load_fused(
        ctx: &GpuContext,
        w: &mut dyn Weights,
        in_f: usize,
        parts: &[(&str, usize)],
    ) -> Self {
        let out_total: usize = parts.iter().map(|(_, o)| o).sum();
        let mut data = vec![0f32; in_f * out_total];
        let mut col = 0;
        for (name, out_f) in parts {
            let m = w.matrix_data(&format!("{name}.weight"), in_f, *out_f); // [in_f, out_f]
            for i in 0..in_f {
                for o in 0..*out_f {
                    data[i * out_total + col + o] = m[i * out_f + o];
                }
            }
            col += out_f;
        }
        Self {
            w: ctx.storage(&data),
            bias: None,
            in_f,
            out_f: out_total,
        }
    }

    pub fn forward(&self, ctx: &GpuContext, x: &wgpu::Buffer, rows: usize) -> wgpu::Buffer {
        let y = matmul::matmul(ctx, x, &self.w, rows, self.in_f, self.out_f);
        match (&self.bias, rows) {
            (Some(b), 1) => attention::add(ctx, &y, b, self.out_f),
            _ => y,
        }
    }

    pub fn out_f(&self) -> usize {
        self.out_f
    }

    pub fn weight(&self) -> &wgpu::Buffer {
        &self.w
    }
}

pub struct Q4Linear {
    scales: wgpu::Buffer,
    quants: wgpu::Buffer,
    in_f: usize,
    out_f: usize,
}

impl Q4Linear {
    pub fn from_f32(ctx: &GpuContext, data: &[f32], out_f: usize, in_f: usize) -> Self {
        let (scales, quants) = q4::quantize_q4_0(data, out_f, in_f);
        Self {
            scales: ctx.storage(&scales),
            quants: ctx.storage_u32(&quants),
            in_f,
            out_f,
        }
    }

    pub fn forward(&self, ctx: &GpuContext, x: &wgpu::Buffer) -> wgpu::Buffer {
        q4::gemv(ctx, x, &self.scales, &self.quants, self.in_f, self.out_f)
    }
}

pub enum Proj {
    Dense(Linear),
    Quant(Q4Linear),
}

impl Proj {
    pub fn load(
        ctx: &GpuContext,
        w: &mut dyn Weights,
        quantize: bool,
        name: &str,
        in_f: usize,
        out_f: usize,
    ) -> Self {
        if quantize {
            let data = w.matrix_raw(&format!("{name}.weight"), out_f, in_f);
            Proj::Quant(Q4Linear::from_f32(ctx, &data, out_f, in_f))
        } else {
            Proj::Dense(Linear::load(ctx, w, name, in_f, out_f))
        }
    }

    pub fn load_fused(
        ctx: &GpuContext,
        w: &mut dyn Weights,
        quantize: bool,
        in_f: usize,
        parts: &[(&str, usize)],
    ) -> Self {
        if quantize {
            let mut data = Vec::new();
            for (name, out_f) in parts {
                data.extend(w.matrix_raw(&format!("{name}.weight"), *out_f, in_f));
            }
            let out_total: usize = parts.iter().map(|(_, o)| o).sum();
            Proj::Quant(Q4Linear::from_f32(ctx, &data, out_total, in_f))
        } else {
            Proj::Dense(Linear::load_fused(ctx, w, in_f, parts))
        }
    }

    pub fn forward(&self, ctx: &GpuContext, x: &wgpu::Buffer) -> wgpu::Buffer {
        match self {
            Proj::Dense(l) => l.forward(ctx, x, 1),
            Proj::Quant(q) => q.forward(ctx, x),
        }
    }

    pub fn dense_weight(&self) -> Option<&wgpu::Buffer> {
        match self {
            Proj::Dense(l) => Some(l.weight()),
            Proj::Quant(_) => None,
        }
    }
}

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
        Self {
            w: g,
            dim,
            eps,
            unit,
        }
    }

    pub fn weight(&self) -> &wgpu::Buffer {
        &self.w
    }
    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn forward(&self, ctx: &GpuContext, x: &wgpu::Buffer, rows: usize) -> wgpu::Buffer {
        if self.unit {
            norm::rmsnorm_unit(ctx, x, &self.w, rows, self.dim, self.eps)
        } else {
            norm::rmsnorm(ctx, x, &self.w, rows, self.dim, self.eps)
        }
    }
}
