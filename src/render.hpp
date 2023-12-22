#pragma once

#include "app.hpp"
#include "model.hpp"
#include "camera.hpp"

#include <vector>

namespace raytracer {
void render_frame(
    App &app,
    const Camera &camera,
    const std::vector<Model> &models,
    sycl::range<2> img_size,
    sycl::image<2> &image
);
}
