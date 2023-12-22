#include "render.hpp"

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

        std::vector<raytracer::Model> models;
        models.emplace_back(app, "./assets/Lantern.glb");


        raytracer::Camera camera(img_size, sycl::float3(15, 10, 15), sycl::float3(0, 10, 0));

        raytracer::render_frame(app, camera, models, img_size, image);
    } catch (sycl::exception const &e) {
        fmt::println("Caught SYCL exception: {}", e.what());
        std::terminate();
    }

    return 0;
}
