//! GPU timestamp profiler — per-kernel wall-clock *on the GPU itself*.
//!
//! The CPU-side cost of recording a token is ~0.3 ms; the time actually goes
//! into GPU kernel execution. A CPU sampling profiler (cargo-flamegraph /
//! samply) therefore can't see the bottleneck. This module times each compute
//! pass on the GPU via a timestamp `QuerySet` (one begin + one end stamp per
//! pass), then aggregates the deltas by kernel label.
//!
//! Usage (one decode token): `ctx.begin_profile()` before the ops →
//! run the token → after the readback, `ctx.report_profile().await` returns
//! `(kernel, count, total_ms)` rows sorted by total. See PROFILING.md.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

/// Captures begin/end GPU timestamps for every compute pass in one frame.
pub struct Profiler {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer, // QUERY_RESOLVE | COPY_SRC
    staging: wgpu::Buffer, // COPY_DST   | MAP_READ
    period_ns: f32,        // GPU ticks → nanoseconds
    capacity: u32,         // timestamp slots (2 per pass)
    next: Cell<u32>,       // next free slot
    labels: RefCell<Vec<&'static str>>, // kernel label per pass
}

impl Profiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, max_passes: u32) -> Self {
        let capacity = max_passes * 2;
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu-profile"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });
        let bytes = capacity as u64 * 8;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            set,
            resolve,
            staging,
            period_ns: queue.get_timestamp_period(),
            capacity,
            next: Cell::new(0),
            labels: RefCell::new(Vec::new()),
        }
    }

    /// Reserve a `(begin, end)` timestamp pair for the next pass. `None` if full.
    pub fn reserve(&self, label: &'static str) -> Option<(u32, u32)> {
        let i = self.next.get();
        if i + 2 > self.capacity {
            return None;
        }
        self.next.set(i + 2);
        self.labels.borrow_mut().push(label);
        Some((i, i + 1))
    }

    pub fn query_set(&self) -> &wgpu::QuerySet {
        &self.set
    }

    /// Record resolve + copy-to-staging into `enc` (call at flush, pre-submit).
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) {
        let n = self.next.get();
        if n == 0 {
            return;
        }
        enc.resolve_query_set(&self.set, 0..n, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.staging, 0, n as u64 * 8);
    }

    /// Map the resolved timestamps and aggregate per kernel label. Returns
    /// `(label, pass_count, total_ms)` sorted by total descending.
    pub async fn report(&self, device: &wgpu::Device) -> Vec<(&'static str, u32, f64)> {
        let n = self.next.get();
        if n == 0 {
            return vec![];
        }
        let slice = self.staging.slice(..n as u64 * 8);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.await.unwrap().unwrap();
        let data = slice.get_mapped_range();
        let ts: &[u64] = bytemuck::cast_slice(&data);

        let labels = self.labels.borrow();
        let mut agg: BTreeMap<&'static str, (u32, f64)> = BTreeMap::new();
        for (pass, &label) in labels.iter().enumerate() {
            let dt = ts[pass * 2 + 1].saturating_sub(ts[pass * 2]) as f64;
            let ms = dt * self.period_ns as f64 / 1.0e6;
            let e = agg.entry(label).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms;
        }
        drop(data);
        self.staging.unmap();

        let mut rows: Vec<_> = agg.into_iter().map(|(l, (c, ms))| (l, c, ms)).collect();
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        rows
    }
}

/// Pretty-print a per-kernel GPU breakdown (what `report` returns).
pub fn print_report(rows: &[(&'static str, u32, f64)]) {
    let total: f64 = rows.iter().map(|r| r.2).sum();
    eprintln!("\n┌─ GPU per-kernel breakdown (one decode token) ─────────────");
    eprintln!("│ {:<16} {:>7} {:>10} {:>9} {:>6}", "kernel", "passes", "total ms", "µs/pass", "%");
    eprintln!("├───────────────────────────────────────────────────────────");
    for (label, count, ms) in rows {
        let per = ms * 1000.0 / *count as f64;
        let pct = if total > 0.0 { ms / total * 100.0 } else { 0.0 };
        eprintln!("│ {label:<16} {count:>7} {ms:>10.3} {per:>9.1} {pct:>5.1}%");
    }
    eprintln!("├───────────────────────────────────────────────────────────");
    eprintln!("│ {:<16} {:>7} {total:>10.3} {:>9} {:>6}", "TOTAL", "", "", "");
    eprintln!("└─ note: buffer copies (KV write, embed gather) are untimed ─\n");
}
