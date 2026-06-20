//! KV cache: per-layer preallocated GPU buffers `[max_seq, kv_dim]`. Each decode
//! step writes the new token's K/V at its position; attention then reads the
//! `pos+1`-length prefix. Bounded memory, no per-token reallocation.

use crate::context::GpuContext;

pub struct KvCache {
    k: Vec<wgpu::Buffer>,
    v: Vec<wgpu::Buffer>,
    kv_dim: usize,
}

impl KvCache {
    pub fn new(ctx: &GpuContext, n_layers: usize, kv_dim: usize, max_seq: usize) -> Self {
        let alloc = || (0..n_layers).map(|_| ctx.empty(max_seq * kv_dim)).collect();
        Self { k: alloc(), v: alloc(), kv_dim }
    }

    /// Write this token's `k`/`v` (each `[kv_dim]`) into layer `layer` at `pos`.
    pub fn write(&self, ctx: &GpuContext, layer: usize, pos: usize, k: &wgpu::Buffer, v: &wgpu::Buffer) {
        ctx.copy(k, 0, &self.k[layer], pos * self.kv_dim, self.kv_dim);
        ctx.copy(v, 0, &self.v[layer], pos * self.kv_dim, self.kv_dim);
    }

    pub fn k(&self, layer: usize) -> &wgpu::Buffer {
        &self.k[layer]
    }
    pub fn v(&self, layer: usize) -> &wgpu::Buffer {
        &self.v[layer]
    }
}
