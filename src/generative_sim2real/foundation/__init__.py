"""Foundation-model comparison: fixed inputs, image-editor clients (Gemini reported), geometry proxy.

Sub-modules are import-light on purpose: selection/prompt/geometry-registration need only numpy +
OpenCV; the editors and the depth proxy import torch/diffusers/google-genai lazily, so each runs in
the venv that has it (external/foundation-venv for editors, external/da3-venv for the DA3 proxy).
"""
