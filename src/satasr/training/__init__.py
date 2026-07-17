"""Training pipeline: Phase 1 full fine-tune then Phase 2 LoRA (design §5).

Both phases depend on the ``satasr.model.Model`` abstraction and the dataset /
real-audio loaders, never on a concrete checkpoint.
"""
