#include <fmt/core.h>

#include <CLI11.hpp>
#include "render.hpp"
#include "render_megakernel.hpp"
#include "render_wavefront.hpp"

int main(int argc, const char *argv[]) {
    CLI::App cli_app{"App description"};

    uint32_t max_depth = 10;
    cli_app.add_option("-d,--max-depth", max_depth, "Max depth");
    uint32_t sample_count = 32;
    cli_app.add_option("-s,--sample-count", sample_count, "Sample count");

    std::string scene_path = "./assets/sponza.glb";
    cli_app.add_option("scene_path", scene_path, "Scene path");

    bool use_wavefront = false;
    cli_app.add_flag("-w,--wavefront", use_wavefront, "Use wavefront renderer");
    bool use_megakernel = false;
    cli_app.add_flag("-m,--megakernel", use_megakernel, "Use megakernel renderer");

    CLI11_PARSE(cli_app, argc, argv);

    if (!use_wavefront && !use_megakernel) {
        use_wavefront = true;
    }

    fmt::println("Loading scene: {}", scene_path);

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

        raytracer::Scene scene(app, scene_path);

        raytracer::Camera camera(
            img_size,
            scene.camera_position,
            scene.camera_direction,
            scene.camera_focal_length
        );

        std::unique_ptr<raytracer::IRenderer> renderer;
        if (use_megakernel) {
            renderer.reset(new raytracer::MegakernelRenderer(
                app, img_size, image, max_depth, sample_count
            ));
        } else if (use_wavefront) {
            renderer.reset(new raytracer::WavefrontRenderer(
                app, img_size, image, max_depth, sample_count
            ));
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
