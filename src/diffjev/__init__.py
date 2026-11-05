"""DiffJev: permutation-equivariant multi-slot co-denoising for decision models.

Package layout (general algorithm is backbone-agnostic; model specifics live in
``diffjev.backbones``):

- ``config``        : run / method configuration dataclasses
- ``packing``       : shared-query + branch packing, Set-RoPE positions
- ``masks``         : lower-layer and top-layer visibility rules
- ``backbones``     : ``BackboneAdapter`` interface + mock / LLaDA / Nemotron adapters
- ``encoder``       : lower-layer branch encoder and top-layer static detail memory
- ``probability``   : structured partition normalisation, task readouts
- ``diffusion``     : forward corruption, mean-field reverse updates, reference sampler
- ``pruning``       : progressive survivor selection with tie retention
- ``denoiser``      : top-layer slot co-denoiser (training views and inference rollouts)
- ``losses``        : L_mask, teacher-free task CE, Brier
- ``model``         : ``DiffJevModel`` wiring everything together
- ``tasks``         : choice / noul / score task adapters
- ``training``      : freezing policy, parameter counts, formal & synthetic training loops
- ``calibration``   : post-hoc temperature scaling on calibration split (full support)
- ``baselines``     : serialized DLM, causal AR (multi-token trie), BM25, BGE/E5, Jev API, Jeva-4B
- ``data``          : synthetic data, OpenJev/ToolRet/JevBench mappers, splits, augmentation, trace
- ``evaluation``    : metrics and runners for Exp 1, Exp 2, Exp 3, Exp 4, and Exp 5
- ``complexity``    : analytic Phi(n,a) cost formulas and operator-shape GFLOPs counters
"""

__version__ = "0.2.0"
