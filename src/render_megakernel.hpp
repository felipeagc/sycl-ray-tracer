#pragma once

#include "render.hpp"

namespace raytracer {
struct MegakernelRenderer : public IRenderer {
    App &app;

    MegakernelRenderer(App &app);

    void render_frame(
        const Camera &camera,
        const Scene &scene,
        sycl::range<2> img_size,
        sycl::image<2> &image
    ) override;
};
} // namespace raytracer
