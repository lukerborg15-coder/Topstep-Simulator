Optional JSON for `python -m v3.cli --pipeline-config config/your_windows.json`.

Schema: `walk_forward` (list of `train` / `test` windows), `holdout`. Same shape as `v3.pipeline_config.load_pipeline_windows`.

`in_sample_sanity` is no longer used; omit it from any JSON config.
