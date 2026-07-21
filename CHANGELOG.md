# Changelog

Notable changes are recorded here. Entries are grouped by release; the `Unreleased`
section collects changes not yet part of a tagged release.

## Unreleased

### Fixed

- **selector (stage-b): low-RAM runtime cache.** The candidate-search runtime cache
  previously copied every candidate tensor's weights into host RAM twice (a full
  original copy plus a working staging buffer) for all tensors up front. For a ~27B
  model this allocated roughly 2x the weights in host RAM and OOM-killed the run on
  machines with ~28 GB RAM at the `caching original runtime tensors` step.

  Originals are now restored from the GPU device snapshot (already captured by the
  loader) instead of a full host copy, and the working staging buffer is allocated
  lazily and released after each patch. This is the **safe variant**: behavior is
  identical (the snapshot is a byte-exact copy), NVFP4 needs no host copy at all, and
  MXFP6_E2M3 keeps the proven host-copy restore because `ggml_cuda_tensor_snapshot`
  does not yet expose an active device pointer for that type. Host RAM is therefore
  bounded to the MXFP6 subset rather than the whole checkpoint.

  A deeper fix that makes MXFP6 use the device snapshot too would require changes in
  `ggml-cuda` (`nvfp4-adv.cu`) and CUDA testing; it is intentionally out of scope
  here to keep this change low-risk.
