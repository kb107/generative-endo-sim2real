"""Image-set metrics for judging how well an appearance LoRA / generated set matches a real domain.

Not part of the frozen C2 registry. Reads RGB only (reference sets go through the project's
role-checked `generation_dataset`, so target_test and C2 are refused). See compare.py."""
