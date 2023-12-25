#include <fmt/core.h>

#include "render.hpp"
#include "render_megakernel.hpp"

int main(int argc, char *argv[]) {
    try {
        raytracer::App app;

        // Calculate viewport size
        sycl::range<2> img_size = sycl::range<2>(1920, 1080);

        // Create image
        uint8_t *image_buf =
            sycl::malloc_shared<uint8_t>(img_size[0] * img_size[1] * 4, app.queue);
        sycl::image<2> image(
            image_buf,
            sycl::image_channel_order::rgba,
            sycl::image_channel_type::unorm_int8,
            img_size
        );

        raytracer::Scene scene(app, "./assets/cornell.glb");

        raytracer::Camera camera(
            img_size,
            scene.camera_position,
            scene.camera_direction,
            scene.camera_focal_length
        );

        std::unique_ptr<raytracer::IRenderer> renderer(
            new raytracer::MegakernelRenderer(app)
        );
        renderer->render_frame(camera, scene, img_size, image);
    } catch (sycl::exception const &e) {
        fmt::println("Caught SYCL exception: {}", e.what());
        std::terminate();
    }

    return 0;
}
