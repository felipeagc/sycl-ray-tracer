#include <fmt/core.h>

#include "render.hpp"
#include "render_megakernel.hpp"
#include "render_wavefront.hpp"

int main(int argc, char *argv[]) {
    try {
        raytracer::App app;

        const std::string renderer_name = (argc == 2) ? argv[1] : "wavefront";

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

        raytracer::Scene scene(app, "./assets/sponza.glb");

        raytracer::Camera camera(
            img_size,
            scene.camera_position,
            scene.camera_direction,
            scene.camera_focal_length
        );

        std::unique_ptr<raytracer::IRenderer> renderer;
        if (renderer_name == "megakernel") {
            renderer.reset(new raytracer::MegakernelRenderer(app, img_size, image));
        } else if (renderer_name == "wavefront") {
            renderer.reset(new raytracer::WavefrontRenderer(app, img_size, image));
        } else {
            throw std::runtime_error("Unknown renderer");
        }

        renderer->render_frame(camera, scene);
    } catch (sycl::exception const &e) {
        fmt::println("Caught SYCL exception: {}", e.what());
        std::terminate();
    }

    return 0;
}
