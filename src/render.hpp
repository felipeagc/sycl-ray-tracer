#pragma once

#include "app.hpp"
#include "scene.hpp"
#include "camera.hpp"
#include "image_manager.hpp"

#include <vector>

namespace raytracer {
struct IRenderer {
    virtual void render_frame(
        const Camera &camera,
        const Scene &scene,
        sycl::range<2> img_size,
        sycl::image<2> &image
    ) = 0;

    virtual ~IRenderer() {};
};
} // namespace raytracer
