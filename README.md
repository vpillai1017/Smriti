# Smriti

Smriti is a Rust-based LLM inference engine experiment focused on one practical question: can reuse-aware KV residency and scheduling improve successful multi-turn serving goodput without sacrificing quality or stability?

This project is currently a design-and-roadmap repository for the Atlas effort: a focused, evidence-driven inference engine prototype targeting a single NVIDIA GPU and a single dense model stack.

## Project status

Status: research and design in progress.

This is not a production inference server, not a general-purpose LLM platform, and not a claim that Rust alone makes inference faster. The work is intentionally scoped and measured-first.

## What Smriti is exploring

- Single-model, single-GPU serving on NVIDIA hardware
- Continuous batching and chunked prefill as required primitives
- Paged KV attention and reusable prefixes where the measurements justify them
- A Rust control plane with CUDA libraries used for the data plane
- A staged, evidence-gated roadmap instead of broad feature sprawl

## Why this project exists

The design work is built around one core idea:

- Reusable KV state can reduce avoidable re-prefill and improve throughput on multi-turn workloads.
- The benefit must be measured against real traces, tuned baselines, and explicit SLOs.
- Scope is kept narrow until the bottleneck is proven and the implementation is validated.

## Architecture summary

The current design is intentionally constrained:

- One exact dense model: Llama-3.1-8B-Instruct, BF16 weights and KV
- One recorded H100 80 GB configuration
- Single process and single GPU
- Request lifecycle built around admission, scheduling, batching, execution, and reconciliation
- Optional host-memory KV tier only if Stage A measurements justify it

## Current scope

In scope for the initial milestone:

- Rust engine + request orchestration
- CUDA executor integration
- Scheduler and KV cache metadata management
- Streaming HTTP frontend for the required chat-completion subset
- Benchmarks against tuned baseline engines

Deferred until justified by measurements:

- MoE and multi-GPU execution
- Tensor or pipeline parallelism
- Quantization
- Router and disaggregation
- Structured output speculation and other advanced composition features

## Repository contents

- `README.md` — project overview
- `Atlas_ Rust LLM Inference Engine — Design Doc & Roadmap.md` — primary design document and staged roadmap
- `LICENSE` — project license

## Design document

The main project design and roadmap are captured here:

[Atlas: Design Doc and Roadmap](./Atlas_ Rust LLM Inference Engine — Design Doc & Roadmap.md)

## Key principles

- Measure before building
- Rust for orchestration and control flow
- Use CUDA libraries for the heavy data-plane work where practical
- Validate feature combinations and reject unsupported ones explicitly
- Treat correctness contracts as a prerequisite to performance work
- Keep cost per successful request and goodput as the leading metrics

## Roadmap direction

The project follows an evidence-gated roadmap:

1. Baseline characterization and bottleneck measurement
2. Intervention testing in an existing engine where feasible
3. Minimal Rust vertical slice
4. Thesis validation with measured KV residency and scheduling gains

## License

This project is licensed under the Apache 2.0 license.

## Notes

This repository is intended to be a focused engineering experiment and a technical design artifact. It is not yet a production-ready inference runtime, and its success depends on measured evidence and disciplined scope control.
