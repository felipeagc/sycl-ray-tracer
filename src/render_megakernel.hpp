#pragma once

#include "render.hpp"

namespace raytracer {
struct MegakernelRenderer : public IRenderer {
    App &app;
    sycl::range<2> img_size;
    sycl::image<2> &image;

    MegakernelRenderer(App &app, sycl::range<2> img_size, sycl::image<2> &image);

    virtual void render_frame(const Camera &camera, const Scene &scene) override;
};
} // namespace raytracer
