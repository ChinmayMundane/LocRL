# Required MORL-Baselines modules

This directory contains the unmodified CAPQL module and its recursive internal
Python dependencies from MORL-Baselines.

Functions within retained modules are kept intact to preserve upstream behavior
and checkpoint compatibility. Python dependencies such as PyTorch, MO-Gymnasium
and pymoo are installed through the root requirements.txt.

Original MIT terms and attribution are in MORL_BASELINES_LICENSE. File checksums
and upstream provenance are in morl_baselines_manifest.json.
