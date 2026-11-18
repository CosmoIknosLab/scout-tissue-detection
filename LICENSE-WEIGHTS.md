# Licence of the released model weights

The source code of this package is licensed under the Apache License 2.0 (`LICENSE`).

The trained SCOUT heads, which the package downloads from the Hugging Face model repository `Cristiandieti/scout-tissue-detection` (also archived on Zenodo, doi:10.5281/zenodo.23664072), are not part of the package and are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0)**:
<https://creativecommons.org/licenses/by-nc/4.0/legalcode>

Copyright 2026 Cristian Tommasino.

Two heads are distributed:
- `audit/final.pt`, the audit head. It is trained on the errors of the classical base detector and does not use any foundation model.
- `fmheads_hoptimus0/final.pt`, the decoder head of SCOUT-open. It is trained on tokens of H-optimus-0 (`bioptimus/H-optimus-0`, Apache-2.0).

The Virchow2 decoder head of SCOUT (the variant reported as SCOUT in the paper) is **not** distributed: it is trained on outputs of the Virchow2 foundation model, whose licence (CC-BY-NC-ND 4.0, `paige-ai/Virchow2` on Hugging Face) restricts models trained on its outputs.

The released files contain only these two heads. They do not include, and this licence does not cover, any third-party weights (H-optimus-0, Virchow2, or any other model). Those must be obtained from their providers and are subject to their own licences.
