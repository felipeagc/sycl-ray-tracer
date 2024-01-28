import os
import subprocess
import re
import itertools

depth_samples = [
    (10, 32),
    (50, 32),
    (100, 32),

    (10, 64),
    (10, 128),
]
scenes = ['./assets/sponza.glb', './assets/minecraft.glb']
renderers = ['-m', '-w']

with open('benchmark.csv', 'w') as f:
    f.write("renderer,depth,samples,scene,time,rays_per_sec,ray_count\n")

combinations = itertools.product(scenes, depth_samples, renderers)
for (scene, (depth, samples), renderer) in combinations:
    print(f"Running benchmark for {renderer} with {samples} samples and {depth} depth")

    for i in range(6):
        print(f"Iteration {i}")
        output = subprocess.check_output([
            "./build/raytracer",
            renderer,
            "-d", str(depth),
            "-s", str(samples),
            scene
        ])
        if i == 0:
            continue

        output = output.decode("utf-8")

        m = re.search(r'Rays/sec: (\d+\.\d+)M', output)
        rays_per_sec = float(m.group(1))

        m = re.search(r'Time measured: (\d+\.\d+) seconds', output)
        time = float(m.group(1))

        m = re.search(r'Total rays: (\d+)', output)
        ray_count = int(m.group(1))

        print(f"({renderer}, {time}, {rays_per_sec}, {ray_count})")

        with open('benchmark.csv', 'a') as f:
            f.write(f"{renderer},{depth},{samples},{scene},{time},{rays_per_sec},{ray_count}\n")
