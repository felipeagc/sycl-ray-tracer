#include <sycl/sycl.hpp>

#include "app.hpp"
#include "formatters.hpp"
#include "image_manager.hpp"

constexpr sycl::uint2 IMAGE_SIZE = {2, 2};
constexpr uint32_t MAX_IMAGES = 128;

int main() {
    raytracer::App app;

    sycl::float4 *dst_data = sycl::malloc_shared<sycl::float4>(MAX_IMAGES, app.queue);

    uint8_t *img_data = sycl::malloc_device<uint8_t>(
        IMAGE_SIZE.x() * IMAGE_SIZE.y() * MAX_IMAGES * 4, app.queue
    );

    std::vector<std::vector<std::array<uint8_t, 4>>> cpu_img_data;
    cpu_img_data.emplace_back(
        IMAGE_SIZE.x() * IMAGE_SIZE.y(),
        std::array<uint8_t, 4>{
            (uint8_t)(0.1f * 255.0f),
            (uint8_t)(0.1f * 255.0f),
            (uint8_t)(0.1f * 255.0f),
            (uint8_t)(0.1f * 255.0f),
        }
    );
    cpu_img_data.emplace_back(
        IMAGE_SIZE.x() * IMAGE_SIZE.y(),
        std::array<uint8_t, 4>{
            (uint8_t)(0.2f * 255.0f),
            (uint8_t)(0.2f * 255.0f),
            (uint8_t)(0.2f * 255.0f),
            (uint8_t)(0.2f * 255.0f),
        }
    );

    for (uint32_t img_index = 0; img_index < cpu_img_data.size(); img_index++) {
        app.queue.submit([&](sycl::handler &cgh) {
            cgh.memcpy(
                img_data + img_index * IMAGE_SIZE.x() * IMAGE_SIZE.y() * 4,
                cpu_img_data[img_index].data(),
                IMAGE_SIZE.x() * IMAGE_SIZE.y() * 4
            );
        });
    }
    app.queue.wait();

    sycl::image<3> albedo_image(
        img_data,
        sycl::image_channel_order::rgba,
        sycl::image_channel_type::unorm_int8,
        sycl::range<3>(IMAGE_SIZE.x(), IMAGE_SIZE.y(), MAX_IMAGES)
    );

    sycl::sampler smpl(
        sycl::coordinate_normalization_mode::normalized,
        sycl::addressing_mode::repeat,
        sycl::filtering_mode::nearest
    );

    app.queue
        .submit([&](sycl::handler &cgh) {
            sycl::accessor<
                sycl::float4,
                2,
                sycl::access::mode::read,
                sycl::access::target::image_array>
                img_access(albedo_image, cgh);

            cgh.single_task<class test>([=]() {
                dst_data[0] = img_access[0].read(sycl::float2(0.0f, 0.0f), smpl);
                dst_data[1] = img_access[1].read(sycl::float2(0.0f, 0.0f), smpl);
            });
        })
        .wait();

    fmt::println("Color[0]: {}", dst_data[0]);
    fmt::println("Color[1]: {}", dst_data[1]);

    return 0;
}
