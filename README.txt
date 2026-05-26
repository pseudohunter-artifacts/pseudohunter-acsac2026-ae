PseudoHunter ACSAC 2026 Artifact Bundle
========================================

This directory is the reviewer-facing artifact bundle for the ACSAC 2026 paper:

  PseudoHunter: Detecting and Localizing Packed Android Payloads via
  Pseudo-Instruction Normality

This bundle is separate from the research working repository. It contains the
code, lightweight tests, paper result summaries, claim verifiers, environment
metadata, and instructions needed for artifact review. It does not contain
private APKs, DEX files, API keys, local logs, virtual environments, or local
agent/tool configuration.

Directory layout
----------------

  artifact/
    Source code, scripts, configs, tests, paper result JSON files, and helper
    tools copied from the working repository by
    scripts/artifact/build_acsac2026_bundle.py.

  infrastructure/
    Environment setup files, including a Dockerfile, dependency notes, and
    constraints for full re-training.

  claims/
    One subdirectory per paper claim. Each claim has claim.txt, run.sh, and an
    expected/ directory. The run.sh scripts validate the claim against shipped
    result JSON files.

  install.sh
    One-command setup script. It creates a local .venv and installs the package
    in editable mode with the dev and metrics extras.

  use.txt
    Reviewer-facing commands, intended use, limitations, and expected runtime.

  license.txt
    License and redistribution notice.

  licenses/
    Full legal texts for Apache-2.0 and CC BY 4.0.

Quick start
-----------

From this directory:

  bash install.sh

If your default `python` is too new for scientific Python wheels, select a
supported interpreter explicitly, for example:

  PYTHON=python3.11 bash install.sh

  bash claims/claim1_lopo_main/run.sh
  bash claims/claim2_path_ablation/run.sh
  bash claims/claim3_hard_benign_repair/run.sh
  bash claims/claim4_smoke_pipeline/run.sh

Expected runtime
----------------

The default claim verifiers are lightweight and should finish within seconds
after installation. They validate the paper's reported metrics from shipped
JSON outputs.

Full re-training is not the default artifact path because it depends on APK
corpora and model checkpoints that are bulky and may not be redistributable.
The artifact instead includes:

  * result JSON files used for the paper tables;
  * code and configs needed to inspect the pipeline;
  * a smoke test path for the core package;
  * manifest/hash metadata for external APK inputs where redistribution is not
    permitted.

Infrastructure
--------------

The quick path runs on a standard Python 3.9+ environment and does not require
GPU access. A Dockerfile is provided for containerized execution:

  docker build -t pseudohunter-artifact -f infrastructure/Dockerfile .
  docker run --rm pseudohunter-artifact

Full model pre-training or LOPO re-training may require CUDA-capable hardware,
external APK corpora, and longer runtimes; those steps are documented as
optional and are not required for the lightweight claims.

Data redistribution policy
--------------------------

This artifact does not redistribute APK, DEX, ZIP, AAB, APKS, model corpus, or
private runtime traces. When external APKs are required, the artifact provides
metadata, hashes, selection rules, and scripts where possible.
