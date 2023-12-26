#pragma once

#include "render.hpp"

namespace raytracer {
struct WavefrontRenderer : public IRenderer {
    App &app;

    WavefrontRenderer(App &app);

    virtual void render_frame(
        const Camera &camera,
        const Scene &scene,
        sycl::range<2> img_size,
        sycl::image<2> &image
    ) override;
};
} // namespace raytracer
