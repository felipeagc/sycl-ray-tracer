#pragma once

#include <stdexcept>
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <embree4/rtcore_geometry.h>

#include "tiny_gltf.h"

#include "app.hpp"

namespace raytracer {

struct Mesh {
    RTCGeometry geom;

    Mesh(RTCGeometry geom) : geom(geom) {}

    Mesh(const Mesh &) = delete;
    Mesh &operator=(const Mesh &) = delete;

    Mesh(Mesh &&other) {
        this->geom = other.geom;
        other.geom = nullptr;
    }

    Mesh &operator=(Mesh &&other) {
        this->geom = other.geom;
        other.geom = nullptr;
        return *this;
    }

    ~Mesh() {
        if (geom) {
            rtcReleaseGeometry(geom);
        }
    }
};

struct Model {
    std::vector<Mesh> meshes;

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    Model(Model &&) = default;
    Model &operator=(Model &&) = default;

    Model(App &app, const std::string &filepath) {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret =
            loader.LoadBinaryFromFile(&model, &err, &warn, filepath.c_str());

        if (!warn.empty()) {
            printf("Warn: %s\n", warn.c_str());
        }

        if (!ret || !err.empty()) {
            throw std::runtime_error("Failed to load .glTF : " + err);
        }

        for (size_t i = 0; i < model.bufferViews.size(); i++) {
            const tinygltf::BufferView &bufferView = model.bufferViews[i];
        }

        for (auto &mesh : model.meshes) {
            for (auto &primitive : mesh.primitives) {
                if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                    throw std::runtime_error("Only triangles are supported.");
                }

                auto &index_accessor = model.accessors[primitive.indices];
                auto &index_buffer_view =
                    model.bufferViews[index_accessor.bufferView];
                auto &index_buffer = model.buffers[index_buffer_view.buffer];
                assert(index_accessor.type == TINYGLTF_TYPE_SCALAR);

                auto &pos_accessor =
                    model.accessors[primitive.attributes["POSITION"]];
                auto &pos_buffer_view =
                    model.bufferViews[pos_accessor.bufferView];
                auto &pos_buffer = model.buffers[pos_buffer_view.buffer];
                assert(pos_accessor.componentType ==
                       TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(pos_accessor.type == TINYGLTF_TYPE_VEC3);

                assert(index_accessor.count % 3 == 0);
                assert(pos_accessor.count % 3 == 0);

                RTCGeometry geom = rtcNewGeometry(app.embree_device,
                                                  RTC_GEOMETRY_TYPE_TRIANGLE);

                //
                // Vertex buffer
                //

                float *source_pos_data = reinterpret_cast<float *>(
                    &pos_buffer.data[pos_buffer_view.byteOffset +
                                     pos_accessor.byteOffset]);

                float *positions = (float *)rtcSetNewGeometryBuffer(
                    geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                    3 * sizeof(float), pos_accessor.count);

                std::cout << "Positions:\n";
                for (size_t i = 0; i < pos_accessor.count; i++) {
                    for (size_t j = 0; j < 3; j++) {
                        positions[i*3+j] = source_pos_data[i*3+j];
                        std::cout << positions[i*3+j] << " ";
                        if (j % 3 == 2) {
                            std::cout << "\n";
                        }
                    }
                }

                //
                // Index buffer
                //

                uint32_t *indices = (uint32_t *)rtcSetNewGeometryBuffer(
                    geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                    3 * sizeof(uint32_t), index_accessor.count / 3);

                switch (index_accessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                        uint8_t *source_index_data =
                            reinterpret_cast<uint8_t *>(
                                &index_buffer
                                     .data[index_buffer_view.byteOffset +
                                           index_accessor.byteOffset]);

                        for (size_t i = 0; i < index_accessor.count; i++) {
                            indices[i] = source_index_data[i];
                        }
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        uint16_t *source_index_data =
                            reinterpret_cast<uint16_t *>(
                                &index_buffer
                                     .data[index_buffer_view.byteOffset +
                                           index_accessor.byteOffset]);

                        for (size_t i = 0; i < index_accessor.count; i++) {
                            indices[i] = source_index_data[i];
                        }
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                        uint32_t *source_index_data =
                            reinterpret_cast<uint32_t *>(
                                &index_buffer
                                     .data[index_buffer_view.byteOffset +
                                           index_accessor.byteOffset]);

                        for (size_t i = 0; i < index_accessor.count; i++) {
                            indices[i] = source_index_data[i];
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error(
                            "Unsupported index component type.");
                }

                std::cout << "Indices:\n";
                for (size_t i = 0; i < index_accessor.count; i++) {
                    std::cout << indices[i] << " ";
                    if (i % 3 == 2) {
                        std::cout << "\n";
                    }
                }

                rtcCommitGeometry(geom);

                this->meshes.emplace_back(geom);

                // std::cout << "Index count: " << index_accessor.count
                //           << std::endl;
                // std::cout << "Index buffer: " << index_accessor.bufferView
                //           << std::endl;
                // std::cout << "Index byteOffset: " <<
                // index_accessor.byteOffset
                //           << std::endl;

                // std::cout << "Pos count: " << pos_accessor.count <<
                // std::endl; std::cout << "Pos buffer: " <<
                // pos_accessor.bufferView
                //           << std::endl;
                // std::cout << "Pos byteOffset: " << pos_accessor.byteOffset
                //           << std::endl;
            }
        }
    }
};

struct Scene {
    RTCScene scene;

    Scene(const Scene &) = delete;
    Scene &operator=(const Scene &) = delete;

    Scene(Scene &&other) {
        this->scene = other.scene;
        other.scene = nullptr;
    }
    Scene &operator=(Scene &&other) {
        this->scene = other.scene;
        other.scene = nullptr;
        return *this;
    }

    Scene(App &app, const std::vector<Model> &models) {
        this->scene = rtcNewScene(app.embree_device);
        for (auto &model : models) {
            for (auto &mesh : model.meshes) {
                rtcAttachGeometry(this->scene, mesh.geom);
            }
        }

        rtcCommitScene(this->scene);
    }

    ~Scene() {
        if (this->scene) {
            rtcReleaseScene(this->scene);
        }
    }
};

}  // namespace raytracer
