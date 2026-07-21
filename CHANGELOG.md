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

  Originals are now restored from a GPU device snapshot instead of a full host copy,
  and the working staging buffer is allocated lazily and released after each patch.
  This keeps the original weights entirely in VRAM, so host RAM is bounded and a
  ~27B run no longer OOM-kills on a ~28 GB RAM machine.

  `ggml_cuda_tensor_snapshot` / `ggml_cuda_tensor_restore` now support MXFP6_E2M3 in
  addition to NVFP4. The NVFP4 path is unchanged (direct device-to-device copy of the
  active device data). For MXFP6_E2M3 the snapshot/restore go through the ggml backend
  buffer: the tensor is read/written via `ggml_backend_tensor_get` / `ggml_backend_tensor_set`
  with a transient host staging copy, while the snapshot itself lives in VRAM. The
  original host-copy fallback in `quantize_binding_ensure_target_bytes` is retained as
  a last resort (e.g. VRAM pressure) but is no longer on the hot path for either type.
