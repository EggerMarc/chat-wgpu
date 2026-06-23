//! `cargo gpubench` — record a **Metal System Trace** of a real generation run
//! and open it in Instruments. Every GPU dispatch is labeled (`q4_gemv`,
//! `attention`, `rmsnorm`, …), so the GPU track shows kernels by name with real
//! device utilization. macOS + full Xcode required.
//!
//!   cargo gpubench                     # default model, 60 tokens
//!   cargo gpubench <model.gguf> <ntok>
//!
//! Every failure prints a concrete reason + how to fix it (see `fail`).

use std::path::Path;
use std::process::Command;

const MODEL: &str = "/tmp/qwen3-test/qwen3-0.6b-q4_0.gguf";
const TOK: &str = "/tmp/qwen3-test/tokenizer.json";
const TRACE: &str = "/tmp/chatwgpu.trace";

fn main() {
    let model = std::env::args().nth(1).unwrap_or_else(|| MODEL.into());
    let ntok = std::env::args().nth(2).unwrap_or_else(|| "60".into());

    // ── preconditions, each with a fix ──
    if !cfg!(target_os = "macos") {
        fail("Metal System Trace needs macOS + Xcode — you're not on macOS.");
    }
    if ntok.parse::<u32>().is_err() {
        fail(&format!("ntokens must be a number, got `{ntok}`.\n  usage: cargo gpubench <model.gguf> <ntokens>"));
    }
    if !Path::new(&model).exists() {
        fail(&format!("model not found: {model}\n  pass one: cargo gpubench <model.gguf> [ntokens]"));
    }
    if !Path::new(TOK).exists() {
        fail(&format!("tokenizer not found: {TOK}\n  put a tokenizer.json there, or edit TOK in examples/bench.rs"));
    }
    if !runs("xcrun", &["--version"]) {
        fail("`xcrun` not found — install the Xcode command line tools: xcode-select --install");
    }
    if !template_available() {
        fail(
            "the 'Metal System Trace' template isn't available.\n  \
             Instruments ships with the FULL Xcode (App Store), not the Command Line Tools alone.\n  \
             After installing Xcode: sudo xcode-select -s /Applications/Xcode.app",
        );
    }

    // ── build the example we trace ──
    step(
        "build the `generate` example",
        Command::new("cargo").args(["build", "--release", "--example", "generate"]),
        "fix the compile errors above, then re-run.",
    );
    let bin = "target/release/examples/generate";
    if !Path::new(bin).exists() {
        fail(&format!("expected {bin} after build, but it's missing."));
    }

    // ── single-token lifetime (printed to this terminal) ──
    // Run generate directly (xctrace doesn't forward the child's stdout) with
    // WGPU_PROFILE = the last decode token, so it prints that token's lifetime:
    // the CPU-record vs GPU-execute split + per-kernel GPU timestamp breakdown,
    // at the deepest context length of this run.
    let n: u32 = ntok.parse().unwrap();
    let pidx = n.saturating_sub(1).to_string();
    eprintln!("\n[gpubench] ── single-token lifetime (decode token {pidx}, seq≈prompt+{pidx}) ──");
    step(
        "profile the token lifetime",
        Command::new(bin)
            .env("WGPU_PROFILE", &pidx)
            .args([&model, TOK, "The capital of France is", &ntok, "--quantize"]),
        "check the model / tokenizer paths.",
    );

    // ── Metal System Trace (the interactive GPU timeline) ──
    let _ = std::fs::remove_dir_all(TRACE);
    eprintln!("[gpubench] recording Metal System Trace → {TRACE}");
    step(
        "record the trace (xctrace)",
        Command::new("xcrun").args([
            "xctrace", "record",
            "--template", "Metal System Trace",
            "--output", TRACE,
            "--launch", "--",
            bin, &model, TOK, "The capital of France is", &ntok, "--quantize",
        ]),
        "common causes: Instruments needs authorization (open Instruments.app once and allow it), \
         or the model/tokenizer paths are wrong (see the generate output above).",
    );
    if !Path::new(TRACE).exists() {
        fail("xctrace exited 0 but wrote no trace — grant Instruments permission (open Instruments.app once) and retry.");
    }

    // ── open ──
    step("open the trace", Command::new("open").arg(TRACE), "open it manually: open /tmp/chatwgpu.trace");
    eprintln!("[gpubench] ✓ opened {TRACE} — GPU track shows each labeled kernel + device utilization");
}

/// Print a concrete failure reason and exit non-zero.
fn fail(reason: &str) -> ! {
    eprintln!("\n[gpubench] ✗ {reason}\n");
    std::process::exit(1);
}

/// Run a build/record/open step; on non-zero exit or spawn error, explain why.
fn step(what: &str, cmd: &mut Command, hint: &str) {
    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => fail(&format!("could not {what} (exit {}).\n  {hint}", s.code().unwrap_or(-1))),
        Err(e) => fail(&format!("could not {what}: {e}\n  {hint}")),
    }
}

/// Does `cmd` exist and run?
fn runs(cmd: &str, args: &[&str]) -> bool {
    Command::new(cmd).args(args).output().is_ok()
}

/// Is the Metal System Trace template installed (i.e. full Xcode present)?
fn template_available() -> bool {
    Command::new("xcrun")
        .args(["xctrace", "list", "templates"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("Metal System Trace"))
        .unwrap_or(false)
}
