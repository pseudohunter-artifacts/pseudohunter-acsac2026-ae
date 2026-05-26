Infrastructure Notes
====================

Recommended reviewer environment
--------------------------------

  * Python 3.9 or newer
  * Linux/macOS shell, or Windows with WSL/Git Bash
  * Optional: Docker for an isolated environment
  * CPU is sufficient for the default quick claim verifiers

Quick local install
-------------------

  bash install.sh

If your default `python` resolves to a pre-release or very new interpreter, use
Python 3.9-3.12 explicitly:

  PYTHON=python3.11 bash install.sh

Docker build
------------

  docker build -t pseudohunter-artifact -f infrastructure/Dockerfile .
  docker run --rm pseudohunter-artifact

Public infrastructure
---------------------

The quick reviewer path should run on generic public infrastructure that offers
Python 3.9+ and outbound package download access, including standard cloud VMs
or public research platforms. No special hardware is required for the default
claim verifiers.

Constraints for full experiments
--------------------------------

Full model pretraining and LOPO re-training are outside the default artifact
path. They may require:

  * CUDA-capable GPU resources;
  * external APK corpora obtained under their original terms;
  * non-redistributed checkpoints or regenerated checkpoints;
  * substantially longer runtime than the quick verifier path.

The artifact includes the experiment scripts and result summaries needed to
inspect the pipeline and validate the shipped paper numbers.
