"""Multimodal (EEG + ECG + limb sEMG) benchmarks on the three C1 encoders.

``prep_common``   what the two preprocessing scripts share: rates, windows,
                  per-modality filtering and normalisation, the HDF5 layout,
                  an EDF reader
``data``          the dataset over that layout
``encoders``      one C1 encoder per (task, modality), sharing the pretrained
                  core across tasks
``model``         summary tokens, RALF and the two baselines, the container
``train``         the entry point: joint training, selection, the test matrix
"""
