#pragma once

#include <embree4/rtcore_geometry.h>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fmt/core.h>

#include "tiny_gltf.h"

#include "util.hpp"
#include "app.hpp"

static_assert(sizeof(glm::vec3) == 3 * sizeof(float));
static_assert(alignof(glm::vec3) == alignof(float));

template <> struct ::fmt::formatter<glm::vec3> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}') throw_format_error("invalid format");
        return it;
    }
    auto format(glm::vec3 t, fmt::format_context &ctx) {
        return fmt::format_to(ctx.out(), "{} {} {}", t[0], t[1], t[2]);
    }
};

template <> struct ::fmt::formatter<glm::mat4> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && *it != '}') throw_format_error("invalid format");
        return it;
    }

    auto format(glm::mat4 m, fmt::format_context &ctx) {
        auto appender = ctx.out();
        for (int i = 0; i < 4; ++i) {
            appender = fmt::format_to(
                appender, "{} {} {} {}\n", m[i][0], m[i][1], m[i][2], m[i][3]
            );
        }
        return appender;
    }
};

namespace raytracer {

struct Primitive {
    glm::vec3 *positions;
    size_t position_count;
    uint32_t *indices;
    uint32_t index_count;
};

struct Mesh {
    std::vector<Primitive> primitives;
};

struct Node {
    std::optional<uint32_t> parent = {};
    std::vector<RTCGeometry> geometries = {};
    glm::vec3 translation{};
    glm::vec3 scale{1.0f};
    glm::quat rotation{};
    glm::mat4 matrix{1.0f};

    Node() = default;

    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    Node(Node &&other) = default;
    Node &operator=(Node &&other) = default;

    glm::mat4 local_matrix() const {
        return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) *
               glm::scale(glm::mat4(1.0f), scale) * matrix;
    }

    ~Node() {
        for (auto geom : this->geometries) {
            if (geom) {
                rtcReleaseGeometry(geom);
            }
        }
    }
};

struct Model {
    std::vector<Node> nodes;
    std::vector<Mesh> meshes;

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    Model(Model &&) = default;
    Model &operator=(Model &&) = default;

    Model(App &app, const std::string &filepath) {
        tinygltf::Model gltf_model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret = loader.LoadBinaryFromFile(&gltf_model, &err, &warn, filepath.c_str());

        if (!warn.empty()) {
            printf("Warn: %s\n", warn.c_str());
        }

        if (!ret || !err.empty()) {
            throw std::runtime_error("Failed to load .glTF : " + err);
        }

        load_primitives(app, gltf_model);

        const tinygltf::Scene &scene =
            gltf_model.scenes[gltf_model.defaultScene > -1 ? gltf_model.defaultScene : 0];
        this->nodes.resize(gltf_model.nodes.size());
        for (size_t i = 0; i < scene.nodes.size(); i++) {
            uint32_t node_index = scene.nodes[i];
            this->load_node(app, gltf_model, scene.nodes[i], {});
        }
    }

    glm::mat4 node_global_matrix(const Node &node) const {
        glm::mat4 m = node.local_matrix();
        std::optional<uint32_t> p = node.parent;
        while (p) {
            const Node &parent = nodes[p.value()];
            m = parent.local_matrix() * m;
            p = parent.parent;
        }
        return m;
    }

    void load_primitives(App &app, const tinygltf::Model &gltf_model) {
        this->meshes.resize(gltf_model.meshes.size());

        for (size_t i = 0; i < gltf_model.meshes.size(); i++) {
            const tinygltf::Mesh &gltf_mesh = gltf_model.meshes[i];
            Mesh &mesh = this->meshes[i];

            mesh.primitives.resize(gltf_mesh.primitives.size());
            for (size_t j = 0; j < gltf_mesh.primitives.size(); j++) {
                const tinygltf::Primitive &gltf_primitive = gltf_mesh.primitives[j];
                Primitive &primitive = mesh.primitives[j];

                // We only work with indices
                bool has_indices = gltf_primitive.indices > -1;
                assert(has_indices);

                // Position attribute is required
                assert(
                    gltf_primitive.attributes.find("POSITION") !=
                    gltf_primitive.attributes.end()
                );

                const tinygltf::Accessor &pos_accessor =
                    gltf_model
                        .accessors[gltf_primitive.attributes.find("POSITION")->second];
                const tinygltf::BufferView &pos_view =
                    gltf_model.bufferViews[pos_accessor.bufferView];
                const float *buffer_pos = reinterpret_cast<const float *>(
                    &(gltf_model.buffers[pos_view.buffer]
                          .data[pos_accessor.byteOffset + pos_view.byteOffset])
                );

                uint32_t vertex_count = static_cast<uint32_t>(pos_accessor.count);
                uint32_t pos_byte_stride =
                    pos_accessor.ByteStride(pos_view)
                        ? (pos_accessor.ByteStride(pos_view) / sizeof(float))
                        : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);

                primitive.position_count = pos_accessor.count;
                primitive.positions = alignedSYCLMallocDeviceReadOnly<glm::vec3>(
                    app.queue, primitive.position_count, 16
                );

                for (size_t v = 0; v < pos_accessor.count; v++) {
                    primitive.positions[v] =
                        glm::make_vec3(&buffer_pos[v * pos_byte_stride]);
                }

                // Index buffer
                const tinygltf::Accessor &indices_accessor =
                    gltf_model.accessors
                        [gltf_primitive.indices > -1 ? gltf_primitive.indices : 0];
                const tinygltf::BufferView &indices_buffer_view =
                    gltf_model.bufferViews[indices_accessor.bufferView];
                const tinygltf::Buffer &indices_buffer =
                    gltf_model.buffers[indices_buffer_view.buffer];
                const void *indices_data_ptr =
                    &(indices_buffer.data
                          [indices_accessor.byteOffset + indices_buffer_view.byteOffset]);

                assert(indices_accessor.count % 3 == 0);
                primitive.index_count = indices_accessor.count;
                primitive.indices = alignedSYCLMallocDeviceReadOnly<uint32_t>(
                    app.queue, primitive.index_count, 16
                );

                switch (indices_accessor.componentType) {
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                    const uint32_t *buf = static_cast<const uint32_t *>(indices_data_ptr);
                    for (size_t index = 0; index < indices_accessor.count; index++) {
                        primitive.indices[index] = buf[index];
                    }
                    break;
                }
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                    const uint16_t *buf = static_cast<const uint16_t *>(indices_data_ptr);
                    for (size_t index = 0; index < indices_accessor.count; index++) {
                        primitive.indices[index] = buf[index];
                    }
                    break;
                }
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                    const uint8_t *buf = static_cast<const uint8_t *>(indices_data_ptr);
                    for (size_t index = 0; index < indices_accessor.count; index++) {
                        primitive.indices[index] = buf[index];
                    }
                    break;
                }
                default:
                    fmt::println(
                        "Index component type {} not supported!",
                        indices_accessor.componentType
                    );
                    assert(0);
                }
            }
        }
    }

    void load_node(
        App &app,
        const tinygltf::Model &gltf_model,
        uint32_t node_index,
        std::optional<uint32_t> parent_index
    ) {
        const tinygltf::Node &gltf_node = gltf_model.nodes[node_index];
        Node &node = this->nodes[node_index];
        node.parent = parent_index;

        if (gltf_node.translation.size() == 3) {
            node.translation = glm::make_vec3(gltf_node.translation.data());
        }
        if (gltf_node.rotation.size() == 4) {
            node.rotation = glm::make_quat(gltf_node.rotation.data());
        }
        if (gltf_node.scale.size() == 3) {
            node.scale = glm::make_vec3(gltf_node.scale.data());
        }
        if (gltf_node.matrix.size() == 16) {
            node.matrix = glm::make_mat4x4(gltf_node.matrix.data());
        }

        // Node with children
        if (gltf_node.children.size() > 0) {
            for (uint32_t child_index : gltf_node.children) {
                load_node(app, gltf_model, child_index, node_index);
            }
        }

        // Node contains mesh data
        if (gltf_node.mesh != -1) {
            const tinygltf::Mesh gltf_mesh = gltf_model.meshes[gltf_node.mesh];
            Mesh &mesh = this->meshes[gltf_node.mesh];

            node.geometries.resize(mesh.primitives.size());
            for (size_t i = 0; i < mesh.primitives.size(); ++i) {
                Primitive &prim = mesh.primitives[i];
                RTCGeometry *geom = &node.geometries[i];
                *geom = rtcNewGeometry(app.embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);

                rtcSetSharedGeometryBuffer(
                    *geom,
                    RTC_BUFFER_TYPE_VERTEX,
                    0,
                    RTC_FORMAT_FLOAT3,
                    prim.positions,
                    0,
                    sizeof(glm::vec3),
                    prim.position_count
                );

                assert(prim.index_count % 3 == 0);
                uint32_t triangle_count = prim.index_count / 3;
                rtcSetSharedGeometryBuffer(
                    *geom,
                    RTC_BUFFER_TYPE_INDEX,
                    0,
                    RTC_FORMAT_UINT3,
                    prim.indices,
                    0,
                    3 * sizeof(uint32_t),
                    triangle_count
                );

                glm::mat4 global_transform = this->node_global_matrix(node);
                fmt::println("Global transform: {}", global_transform);
                rtcSetGeometryTransform(
                    *geom, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &global_transform[0][0]
                );

                rtcCommitGeometry(*geom);
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
        flush(std::cout);
        this->scene = rtcNewScene(app.embree_device);
        for (auto &model : models) {
            for (auto &node : model.nodes) {
                for (auto &geom : node.geometries) {
                    rtcAttachGeometry(this->scene, geom);
                }
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

} // namespace raytracer
