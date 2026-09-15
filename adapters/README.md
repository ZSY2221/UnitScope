# Third-party adapters

Upstream source code is not copied into this repository.

## PER 2025

- Official repository: `https://github.com/JacobMaciejewski/PER-Design-Space-Exploration`
- Commit used: `09eeb4a27dacf41eb6db11d9ae898b4ac3e17bbd`.
- Prepare FEBRL1--3 with `per2025/prepare_febrl.py`, then pass the
  checkout path to `per2025/run_per_febrl.py --official-repo <checkout>`.

The run manifest records the upstream commit.

## ULDP-FL

- Official repository: `https://github.com/FumiyukiKato/uldp-fl`
- Commit used: `d820e24f5c3cc7275a2b1e78111a46c0da1e8ebf`
- `uldp_fl/run_official_uldp_native.ps1` checks the repository and runs the
  native configuration. Set its checkout path before running it.
