# SeizeIT2 multimodal figures: ralf_frozen_szo

Drawn by `scripts/visualize_seizeit2.py` at commit `0e39ecc0bd77b857c14a7212d2d32c53d0f9d7c8` from the synced results of `ralf_frozen_szo` (branch `cineca-results`, snapshot 2026-10-26 14:58), the five frozen RALF variants of the schedule sweep, `weighted_logit_frozen_szo`, `concat_frozen_szo` and the eight EEG-alone runs. Vector figures with editable text; the arrays behind each are in `figure_data/` (not committed), what produced each in `figure_metadata/`.

## fig_main

SeizeIT2, official split, 29 test patients (2,912 h, 181 seizures). (a), (b): the dataset paper's event scoring (any-overlap, recording means), the 11 FA/h line is the SVM's operating point; each EEG-alone encoder in (a) at its better of frozen / fine-tuned; (d): the window-level test set, the metric the runs are selected on; the RALF band in (a), (d) is the min-max over its 6 frozen training variants (the schedule sweep), the line the original configuration; (c) corruption severity 0.7

![fig_main](figures/fig_main.png)

