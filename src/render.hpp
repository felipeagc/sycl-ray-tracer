#pragma once

#include "app.hpp"
#include "scene.hpp"
#include "camera.hpp"
#include "image_manager.hpp"

#include <vector>

namespace raytracer {

constexpr uint32_t MAX_DEPTH = 10;
constexpr uint32_t SAMPLE_COUNT = 256;

struct IRenderer {
    virtual void render_frame(
        const Camera &camera,
        const Scene &scene
    ) = 0;

    virtual ~IRenderer() {};
};
} // namespace raytracer
