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

- **selector (stage-b): bound per-policy patch RAM.** `patch_policy_once` previously
  quantized every tensor in a policy in parallel and only then copied the candidates in,
  so all candidate staging buffers (`working_target_bytes`) were live simultaneously.
  For large candidate types (BF16/F16/Q8_0) this peaked at tens of GB of host RAM and
  thrashed swap on small-RAM machines during the first real policy after the baseline.
  Candidate staging is now applied (copy-in + free) per tensor, and the parallel path
  quantizes and applies one window of tensors at a time. Peak host RAM is now bounded by
  roughly `PATCH_WINDOW` (default 16, tunable via `SELECTOR_PATCH_WINDOW`) times the
  largest single candidate tensor, instead of the whole policy at once. Behavior and
  restore-on-failure semantics are unchanged.

- **selector (stage-b): optional GGUF-based MXFP6 restore.** When VRAM is nearly full
  after the model is loaded, the MXFP6 device snapshot allocation can fail and the
  old code fell back to a per-tensor host copy (`original_target_bytes`), which for
  hundreds of MXFP6 tensors summed to tens of GiB of host anonymous memory. With
  `SELECTOR_MXFP6_FILE_RESTORE=1` the fallback reads the original tensor data from
  the GGUF checkpoint file at restore time instead of keeping a persistent host copy,
  eliminating the large host-RAM overhead. Restore reads are slower (SSD pread per
  tensor instead of memcpy from RAM), but `restore_all()` only runs between policy
  evaluations and on failure, not per chunk. Default is `0` (old behavior: host copy).
  Enable on machines with very limited RAM where the host copy would cause swapping.
