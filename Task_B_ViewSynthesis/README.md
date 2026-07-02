# Task B — Multi-Camera Novel-View Synthesis

## Task

Reconstruct what one camera on a self-driving car saw at an **intermediate moment**,
given the surrounding frames in time plus a dense LiDAR point cloud. Scored by **PSNR**
against the ground-truth frame.

- 6-camera rig: `front, left_fwd, left_bwd, right_fwd, right_bwd, rear`
- Inputs: all cameras at **t0** and **t1** (2 s apart), a target camera + target time in
  between, camera intrinsics/extrinsics, and ~10M LiDAR points in world coordinates.
- Output: the target camera's image at the target time (a frame that was never recorded).

## Solution

Blend three independent estimates of the target frame, weighted by confidence:

1. **Temporal optical-flow interpolation** — DIS flow both ways between t0/t1, warp toward
   the target time, occlusion-aware blend.
2. **RIFE** — deep frame interpolation, mixed in (`rife_mix=0.42`).
3. **LiDAR image-based rendering** — project the LiDAR cloud into the target view to get a
   dense depth map, back-project each pixel to 3D, re-project into a source camera and
   sample its color (`lidar_mix=0.30`).

Output written as near-lossless JPEG (the metric is pixel-accurate PSNR).

## Files

| File | Description |
|---|---|
| `solution.py` | Full pipeline: optical flow, RIFE, LiDAR projection & blending |
| `generate_submission.py` | Runs the pipeline over the dataset, packs the submission |
