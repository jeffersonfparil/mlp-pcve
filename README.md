# mlp-pcve
Multi-layer perceptrons for phenotype and covariance estimations

## Project Overview

mlp-pcve implements a Multi-Layer Perceptron (MLP) from scratch. It serves as a data-driven alternative to traditional Mixed Linear Models (MLMs). The core goal is to replace the need to manually specify complex covariance structures with a neural network that learns the covariance structures implicitly from the data.

## Goals

The ultimate objective is to obtain more accurate and robust estimates for the key components of a field trial:

| Component | Description | Traditional MLM Role | MLP Approach |
|---|---|---|---|
| Entry effects | The genetic or varietal performance. | Fixed or Random Effect (BLUEs/BLUPs) | Learned non-linear function of entry ID/Factors. |
| Treatment effects | The impact of experimental applications (e.g., fertiliser, spacing). | Fixed Effect | Learned non-linear function of treatment inputs. |
| Spatial effects | The local, non-genetic variation within the field. | Residual Covariance (R-Matrix, e.g., AR(1), SP(exp)) | Learned from plot coordinates (x, y) as input features. |
| Year and seasonal effects | Variation due to time, environment, or growing season. | Fixed or Random Effect | Learned non-linear function of seasonal/environmental covariates. |

## Current Focus: Rust

Rust is the primary development language. The implementation focuses on:
 * Memory Safety: Leveraging Rust's ownership model for safe and correct matrix operations.
 * Performance: Achieving high performance for matrix multiplication and gradient calculation, essential for deep learning operations.
 * Low-Level Control: The entire backpropagation algorithm and optimizer logic (e.g., Stochastic Gradient Descent) are being built manually.

## Prototype and Future Plans
| Language | Status | Role | Rationale |
|---|---|---|---|
| Julia | Prototype Complete | Initial mathematical and algorithmic validation. | Excellent native support for high-speed numerical computing and quick iteration. |
| Rust | Active Development | High-performance, production-grade core implementation. | Safety, speed, and concurrency for resource-intensive operations. |
| Zig | Planned | Potential future port or alternative high-performance core. | Low-level control, simple C interoperability, and fine-grained memory management. |
