# Plan

TODO:
- [x] Load vertex normal attributes
- [x] Use hit (u, v) to interpolate normals (to get smooth normals)
- [x] Light emittance
- [ ] Load texture UVs
- [ ] Load textures
- [ ] Sample texture colors

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
