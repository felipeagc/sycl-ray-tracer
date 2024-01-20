# Plan

TODO:
- [x] Load vertex normal attributes
- [x] Use hit (u, v) to interpolate normals (to get smooth normals)
- [x] Light emittance
- [x] Load texture UVs
- [x] Load textures
- [x] Sample texture colors

- [x] Add cli args parsing for specifying max depth and sample count
- [x] Use specialization constants to pass max depth and sample count to kernels

- [x] Buffer that stores rng state for each pixel
      This will help avoid patterns caused by bad RNG
- [x] Optimize space taken by ray data
    - [x] Try float16 for direction vectors
- [x] Avoid that global atomic (probably biggest bottleneck)
      Use group shared memory for storing produced rays and then combine them later.

- [x] Avoid global atomic in generate_camera_rays

- [ ] Russian roulette ray tracing
      https://computergraphics.stackexchange.com/questions/2316/is-russian-roulette-really-the-answer
      https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting
- [ ] Use splats
      https://aras-p.info/blog/2018/04/25/Daily-Pathtracer-12-GPU-Buffer-Oriented-D3D11/

## Performance log

### Sponza scene

- Params: (d=10, s=32)
    Megakernel:
    - 2023-12-26 19:00 - 1.11s - 545M rays/s
    - 2023-12-27 14:33 - Ray data compaction? - 1.01s - 599.89M rays/s
    - 2023-12-27 16:29 - ??? - 0.96s - 634.38M rays/s
    - 2024-01-20 14:05 - Back from vacation - 0.91s - 665.08M rays/s

    Wavefront:
    - 2023-12-26 19:00 - Initial implementation - 32.89s - 18.57M rays/s
    - 2023-12-26 22:19 - RNG buffer - 33.26s - 18.36M rays/s
    - 2023-12-27 14:33 - Stop passing sycl::stream to kernel - 1.93s - 315.43M rays/s
    - 2023-12-27 14:40 - Decrease shoot_rays group size from 512 to 32 - 1.57s - 388.18M rays/s
    - 2023-12-27 14:52 - Remove global atomic from generate_camera_rays - 1.56s - 390.85M rays/s
    - 2023-12-27 15:03 - Decrease shoot_rays group size from 32 to 16 - 1.43s - 427.37M rays/s
    - 2024-01-20 14:05 - Back from vacation - 1.36s - 447.65M rays/s

- Params: (d=20, s=256)
    Megakernel:
    - 2023-12-27 14:52 - 15.48s - 18.03M rays/s

    Wavefront:
    - 2023-12-27 14:52 - 23.26s - 12.01M rays/s
    - 2023-12-27 15:03 - 20.54s - 13.60M rays/s

## Wavefront raytracing

Consists of 4 phases:
1. **Generate:** kernel that generates the primary rays for each pixel and stores them in a buffer.
2. **Extend:** intersects the rays produced by phase 1 with the scene, storing the intersection results in a buffer.
3. **Shade:** take the intersection results from phase 2 and evaluates the shading mode.
   This step may or may not generate new rays, depending on whether a path was terminated or not.
   A path that spawns a new ray writes the new ray to a buffer.
   Paths that directly sample light sources ("explicit light sampling") write a shadow ray to a second buffer.
4. **Connect:** traces the shadow rays generated in phase 3. It is similar to phase 2, but instead in this step
   we only need to find *any* intersection, not the *nearest* one.
   After phase 4, we are left with buffers that we use again in phase 2, until there are no more rays left or we
   reach a maximum of iterations.

Buffers:
1. Buffer for primary rays.
2. Buffer for intersection results of phase 2.
3. Buffer for path extensions output by phase 3.
4. Buffer for shadow rays output by phase 3.
