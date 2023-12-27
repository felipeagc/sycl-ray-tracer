#pragma once

#include "render.hpp"

namespace raytracer {
struct MegakernelRenderer : public IRenderer {
    App &app;
    sycl::range<2> img_size;
    sycl::image<2> &image;
    const uint32_t max_depth;
    const uint32_t sample_count;

    MegakernelRenderer(
        App &app,
        sycl::range<2> img_size,
        sycl::image<2> &image,
        uint32_t max_depth,
        uint32_t sample_count
    );

    virtual void render_frame(const Camera &camera, const Scene &scene) override;
};
} // namespace raytracer
