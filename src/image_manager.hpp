#pragma once

#include <sycl/sycl.hpp>
#include <fmt/core.h>

#include "app.hpp"
#include "stb_image_resize2.h"

namespace raytracer {

constexpr size_t MAX_IMAGES = 128;
constexpr size_t IMAGE_CHANNELS = 4;
constexpr sycl::int2 IMAGE_SIZE = {512, 512};

using ImageReadAccessor = sycl::accessor<
    sycl::float4,
    2,
    sycl::access::mode::read,
    sycl::access::target::image_array>;

struct Image {
    std::vector<uint8_t> data;
};

struct ImageRef {
    uint32_t index;
};

struct ImageManager {
    std::vector<Image> images;

    ImageManager(const ImageManager &) = delete;
    ImageManager &operator=(const ImageManager &) = delete;

    ImageManager(ImageManager &&) = delete;
    ImageManager &operator=(ImageManager &&) = delete;

    ImageRef upload_image_pixel(sycl::float4 color) {
        uint32_t image_index = this->images.size();
        if (image_index >= MAX_IMAGES) {
            fmt::print("Too many images uploaded\n");
            std::terminate();
        }

        this->images.push_back(Image{
            .data =
                std::vector<uint8_t>(IMAGE_SIZE.x() * IMAGE_SIZE.y() * IMAGE_CHANNELS),
        });

        for (uint32_t i = 0; i < IMAGE_SIZE.x() * IMAGE_SIZE.y(); i++) {
            for (uint32_t j = 0; j < IMAGE_CHANNELS; j++) {
                this->images[image_index].data[i * IMAGE_CHANNELS + j] =
                    (uint8_t)(color[j] * 255.0f);
            }
        }

        return ImageRef{image_index};
    }

    ImageRef upload_image(uint32_t width, uint32_t height, const uint8_t *data) {
        uint32_t image_index = this->images.size();
        if (image_index >= MAX_IMAGES) {
            fmt::print("Too many images uploaded\n");
            std::terminate();
        }

        this->images.push_back(Image{
            .data =
                std::vector<uint8_t>(IMAGE_SIZE.x() * IMAGE_SIZE.y() * IMAGE_CHANNELS),
        });

        uint8_t *output = stbir_resize_uint8_srgb(
            data,
            width,
            height,
            0,
            this->images[image_index].data.data(),
            IMAGE_SIZE.x(),
            IMAGE_SIZE.y(),
            0,
            STBIR_RGBA
        );
        assert(output == this->images[image_index].data.data());

        fmt::println(
            "Resized image {} from {}x{} to {}x{}",
            image_index,
            width,
            height,
            IMAGE_SIZE.x(),
            IMAGE_SIZE.y()
        );

        return ImageRef{image_index};
    }

    sycl::image<3> bake_image(sycl::queue &q) {
        uint8_t *img_data = sycl::malloc_device<uint8_t>(
            IMAGE_SIZE.x() * IMAGE_SIZE.y() * IMAGE_CHANNELS * MAX_IMAGES, q
        );

        for (uint32_t img_index = 0; img_index < this->images.size(); img_index++) {
            q.submit([&](sycl::handler &cgh) {
                cgh.memcpy(
                    img_data +
                        img_index * IMAGE_SIZE.x() * IMAGE_SIZE.y() * IMAGE_CHANNELS,
                    this->images[img_index].data.data(),
                    IMAGE_SIZE.x() * IMAGE_SIZE.y() * IMAGE_CHANNELS
                );
            });
        }
        q.wait();

        sycl::image<3> baked_image(
            img_data,
            sycl::image_channel_order::rgba,
            sycl::image_channel_type::unorm_int8,
            sycl::range<3>(IMAGE_SIZE.x(), IMAGE_SIZE.y(), MAX_IMAGES)
        );

        fmt::println("Baked {} images into array", this->images.size());

        this->images.clear();

        return baked_image;
    }
};
} // namespace raytracer
