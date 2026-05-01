Optional JSON for `python -m v3.cli --pipeline-config config/your_windows.json`.

Schema: `in_sample_sanity` (`name`, `start`, `end`), `walk_forward` (list of `train` / `test` windows), `holdout`. Same shape as `v3.pipeline_config.load_pipeline_windows`.
