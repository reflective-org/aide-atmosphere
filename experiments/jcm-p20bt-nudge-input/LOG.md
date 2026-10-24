# Log: jcm-p20bt-nudge-input

## 2026-10-24

- Reframe decided with Shreyas (see decisions/). The generator, the ERA5-target zarr, the physics nudge, the dataset/trainer plumbing, the overfit and the smoke run are all done (see EXPERIMENT.md, Ladder record).
- The first launch on GPUs 2-7 was stopped during setup: susanne holds GPU 7. Relaunched on GPUs 2-6. A gate on GPU 1 was refused by the permission classifier, since GPU 1 is susanne's, so evals run on GPU 6 alongside training.
