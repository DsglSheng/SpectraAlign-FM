# Downstream fine-tuning

The [task registry](configs/downstream_tasks_v1.json) defines classification, detection, change detection, segmentation, unmixing, denoising, and inversion tasks. The [task configurations](configs/tasks/) contain the hyperparameters for the tasks that use a single training script. Inversion workflows specify their prepared datasets and multiple command steps in the registry.

## Inspect a task

Run the following from the repository root. Set `SPECTRAALIGN_DATA_ROOT` to the absolute path to your datasets and prototype banks. Each task configuration lists its expected data location. Update those paths if your dataset layout differs. For example, `classification_hc` expects its files under `Aware_open/data/classification/hanchuan` relative to the data root.

```bash
python downstream/scripts/run_task_from_config.py --list
python downstream/scripts/run_task_from_config.py --check --allow-missing-checkpoint
python downstream/scripts/run_task_from_config.py --task classification_hc --print-command
```

## Fine-tune with a checkpoint

The runner looks for the Stage B checkpoint produced by the joint recipe. Supply `--checkpoint` or set `AWARE_DZ1_CKPT` to use a different local checkpoint. `PROTO_EO1` and `PROTO_GF5` override individual bank paths.

```bash
python downstream/scripts/run_task_from_config.py \
  --task classification_hc \
  --checkpoint /path/to/stage-b-checkpoint.pt
```

Outputs are written under `downstream/results/` unless `DZ1_DOWNSTREAM_ROOT` is set. The preview reports the exact invocation and output path. Tasks using a third-party baseline require its source and model files, as indicated by the task arguments.
