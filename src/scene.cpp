#include "scene.hpp"

#include <embree4/rtcore_geometry.h>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <glm/gtc/type_ptr.hpp>
#include <fmt/core.h>

#include "formatters.hpp"
#include "util.hpp"

static_assert(sizeof(glm::vec3) == 3 * sizeof(float));
static_assert(alignof(glm::vec3) == alignof(float));

namespace raytracer {
glm::mat4 Node::local_matrix() const {
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) *
           glm::scale(glm::mat4(1.0f), scale) * matrix;
}

Node::~Node() {
    for (auto geom : this->geometries) {
        if (geom) {
            rtcReleaseGeometry(geom);
        }
    }
}

// Scene::Scene(Scene &&other) {
//     this->scene = other.scene;
//     other.scene = nullptr;

//     this->global_scale = other.global_scale;

//     this->nodes = std::move(other.nodes);
//     this->meshes = std::move(other.meshes);
// }

// Scene &Scene::operator=(Scene &&other) {
//     this->scene = other.scene;
//     other.scene = nullptr;

//     this->global_scale = other.global_scale;

//     this->nodes = std::move(other.nodes);
//     this->meshes = std::move(other.meshes);
//     this->image_manager = std::move(other.image_manager);

//     return *this;
// }

Scene::Scene(App &app, const std::string &filepath, glm::vec3 global_scale)
    : global_scale(global_scale) {
    tinygltf::Model gltf_model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    loader.SetStoreOriginalJSONForExtrasAndExtensions(true);
    bool ret = loader.LoadBinaryFromFile(&gltf_model, &err, &warn, filepath.c_str());

    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!ret || !err.empty()) {
        throw std::runtime_error("Failed to load .glTF : " + err);
    }

    load_images(app, gltf_model);
    this->image_array = this->image_baker.bake_image(app.queue);

    load_primitives(app, gltf_model);

    const tinygltf::Scene &scene =
        gltf_model.scenes[gltf_model.defaultScene > -1 ? gltf_model.defaultScene : 0];

    if (auto sky_color = scene.extras.Get("sky_color");
        sky_color.IsArray() && sky_color.Size() == 3) {
        this->sky_color = sycl::float3(
            (float)sky_color.Get(0).GetNumberAsDouble(),
            (float)sky_color.Get(1).GetNumberAsDouble(),
            (float)sky_color.Get(2).GetNumberAsDouble()
        );
        fmt::println("Sky color: {}", this->sky_color);
    }

    if (auto sky_strength = scene.extras.Get("sky_strength"); sky_strength.IsNumber()) {
        float sky_strength_float = (float)sky_strength.GetNumberAsDouble();
        this->sky_color *= sky_strength_float;
        fmt::println("Sky strength: {}", sky_strength_float);
    }

    this->nodes.resize(gltf_model.nodes.size());
    for (uint32_t node_index : scene.nodes) {
        this->load_node(app, gltf_model, node_index, {});
    }

    this->scene = rtcNewScene(app.embree_device);
    for (auto &node : this->nodes) {
        for (auto &geom : node.geometries) {
            rtcAttachGeometry(this->scene, geom);
        }
    }
    rtcCommitScene(this->scene);

    if (this->camera_node_index) {
        auto &gltf_camera_node = gltf_model.nodes[this->camera_node_index];
        Node &camera_node = this->nodes[this->camera_node_index];
        glm::mat4 camera_transform = this->node_global_matrix(camera_node);

        this->camera_position = glm::vec3(camera_transform[3]);

        glm::quat camera_rotation = glm::quat_cast(camera_transform);

        glm::vec3 euler = glm::degrees(glm::eulerAngles(camera_rotation));

        glm::vec3 forward_vector = glm::vec3(0.0f, 0.0f, -1.0f);
        this->camera_direction = glm::normalize(camera_rotation * forward_vector);

        float yfov = gltf_model.cameras[gltf_camera_node.camera].perspective.yfov;
        float aspect_ratio =
            gltf_model.cameras[gltf_camera_node.camera].perspective.aspectRatio;

        this->camera_focal_length = 1.0f / glm::tan(yfov / 2.0f);
    }
}

Scene::~Scene() {
    if (this->scene) {
        rtcReleaseScene(this->scene);
    }
}

glm::mat4 Scene::node_global_matrix(const Node &node) const {
    glm::mat4 m = node.local_matrix() * glm::scale(glm::mat4(1.0f), this->global_scale);
    std::optional<uint32_t> p = node.parent;
    while (p) {
        const Node &parent = nodes[p.value()];
        m = parent.local_matrix() * m;
        p = parent.parent;
    }
    return m;
}

void Scene::load_images(App &app, const tinygltf::Model &gltf_model) {
    this->images.resize(gltf_model.images.size());

    fmt::println("Loading {} images", gltf_model.images.size());

    for (size_t i = 0; i < gltf_model.images.size(); i++) {
        const tinygltf::Image &gltf_image = gltf_model.images[i];

        assert(!gltf_image.as_is);

        this->images[i] = this->image_baker.upload_image(
            gltf_image.width, gltf_image.height, gltf_image.image.data()
        );
    }
}

void Scene::load_primitives(App &app, const tinygltf::Model &gltf_model) {
    this->meshes.resize(gltf_model.meshes.size());

    for (size_t i = 0; i < gltf_model.meshes.size(); i++) {
        const tinygltf::Mesh &gltf_mesh = gltf_model.meshes[i];
        Mesh &mesh = this->meshes[i];

        mesh.primitives.resize(gltf_mesh.primitives.size());
        for (size_t j = 0; j < gltf_mesh.primitives.size(); j++) {
            const tinygltf::Primitive &gltf_primitive = gltf_mesh.primitives[j];
            Primitive &primitive = mesh.primitives[j];

            assert(gltf_primitive.material > -1);

            const tinygltf::Material &gltf_material =
                gltf_model.materials[gltf_primitive.material];

#if 1
            fmt::println("Material[{}]: {}", gltf_primitive.material, gltf_material.name);
            for (auto &ext : gltf_material.extensions) {
                fmt::println("Extension: {}", ext.first);
            }
#endif

            const auto &pbr = gltf_material.pbrMetallicRoughness;
            const auto &base_color_vec = pbr.baseColorFactor;
            sycl::float3 base_color =
                sycl::float3(base_color_vec[0], base_color_vec[1], base_color_vec[2]);

            const auto &emissive_vec =
                gltf_model.materials[gltf_primitive.material].emissiveFactor;
            sycl::float3 emissive =
                sycl::float3(emissive_vec[0], emissive_vec[1], emissive_vec[2]);

            float emissive_strength = 0.0f;
            if (auto emissive_strength_ext =
                    gltf_material.extensions.find("KHR_materials_emissive_strength");
                emissive_strength_ext != gltf_material.extensions.end()) {
                emissive_strength =
                    (float)emissive_strength_ext->second.Get("emissiveStrength")
                        .GetNumberAsDouble();
            }
            emissive = emissive * emissive_strength;

            auto ior_ext = gltf_material.extensions.find("KHR_materials_ior");
            auto transmission_ext =
                gltf_material.extensions.find("KHR_materials_transmission");

            if (ior_ext != gltf_material.extensions.end() &&
                transmission_ext != gltf_material.extensions.end()) {
                float ior = (float)ior_ext->second.Get("ior").GetNumberAsDouble();
                primitive.material = MaterialDielectric{
                    .ior = ior,
                };
                fmt::println("Dielectric: ior={}", ior);
            } else if (pbr.metallicFactor > 0.01f) {
                Texture texture = Texture(base_color);
                if (gltf_material.pbrMetallicRoughness.baseColorTexture.index > -1) {
                    uint32_t texture_index =
                        gltf_material.pbrMetallicRoughness.baseColorTexture.index;
                    uint32_t image_index = gltf_model.textures[texture_index].source;

                    texture = Texture(this->images[image_index]);
                }

                primitive.material = MaterialMetallic{
                    .albedo = texture,
                    .roughness = (float)pbr.roughnessFactor,
                    .emissive = emissive,
                };
                fmt::println(
                    "Metallic: roughness={}, emissive={}",
                    (float)pbr.roughnessFactor,
                    emissive
                );
            } else {
                Texture texture = Texture(base_color);
                if (gltf_material.pbrMetallicRoughness.baseColorTexture.index > -1) {
                    uint32_t texture_index =
                        gltf_material.pbrMetallicRoughness.baseColorTexture.index;
                    uint32_t image_index = gltf_model.textures[texture_index].source;

                    texture = Texture(this->images[image_index]);
                }

                primitive.material = MaterialDiffuse{
                    .albedo = texture,
                    .emissive = emissive,
                };
                fmt::println("Diffuse: albedo={}, emissive={}", base_color, emissive);
            }

            // We only work with indices
            bool has_indices = gltf_primitive.indices > -1;
            assert(has_indices);

            // Position attribute is required
            assert(
                gltf_primitive.attributes.find("POSITION") !=
                gltf_primitive.attributes.end()
            );

            // Normal attribute is required
            assert(
                gltf_primitive.attributes.find("NORMAL") !=
                gltf_primitive.attributes.end()
            );

            // UV attribute is required
            assert(
                gltf_primitive.attributes.find("TEXCOORD_0") !=
                gltf_primitive.attributes.end()
            );

            // Position buffer

            const tinygltf::Accessor &pos_accessor =
                gltf_model.accessors[gltf_primitive.attributes.find("POSITION")->second];
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

            primitive.vertex_count = pos_accessor.count;
            primitive.positions = alignedSYCLMallocDeviceReadOnly<glm::vec3>(
                app.queue, primitive.vertex_count, 16
            );

            for (size_t v = 0; v < pos_accessor.count; v++) {
                primitive.positions[v] = glm::make_vec3(&buffer_pos[v * pos_byte_stride]);
            }

            // Normal buffer

            const tinygltf::Accessor &normal_accessor =
                gltf_model.accessors[gltf_primitive.attributes.find("NORMAL")->second];
            const tinygltf::BufferView &normal_view =
                gltf_model.bufferViews[normal_accessor.bufferView];
            const float *buffer_normal = reinterpret_cast<const float *>(
                &(gltf_model.buffers[normal_view.buffer]
                      .data[normal_accessor.byteOffset + normal_view.byteOffset])
            );

            uint32_t normal_byte_stride =
                normal_accessor.ByteStride(normal_view)
                    ? (normal_accessor.ByteStride(normal_view) / sizeof(float))
                    : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);

            primitive.normals = alignedSYCLMallocDeviceReadOnly<glm::vec3>(
                app.queue, primitive.vertex_count, 16
            );

            for (size_t v = 0; v < primitive.vertex_count; v++) {
                primitive.normals[v] =
                    glm::make_vec3(&buffer_normal[v * normal_byte_stride]);
            }

            // UV buffer

            const tinygltf::Accessor &uv_accessor =
                gltf_model
                    .accessors[gltf_primitive.attributes.find("TEXCOORD_0")->second];
            const tinygltf::BufferView &uv_view =
                gltf_model.bufferViews[uv_accessor.bufferView];
            const float *buffer_uv = reinterpret_cast<const float *>(
                &(gltf_model.buffers[uv_view.buffer]
                      .data[uv_accessor.byteOffset + uv_view.byteOffset])
            );

            uint32_t uv_byte_stride =
                uv_accessor.ByteStride(uv_view)
                    ? (uv_accessor.ByteStride(uv_view) / sizeof(float))
                    : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);

            primitive.uvs = alignedSYCLMallocDeviceReadOnly<sycl::float2>(
                app.queue, primitive.vertex_count, 16
            );

            for (size_t v = 0; v < primitive.vertex_count; v++) {
                primitive.uvs[v] = *((sycl::float2 *)&buffer_uv[v * uv_byte_stride]);
            }

            // Index buffer
            const tinygltf::Accessor &indices_accessor =
                gltf_model
                    .accessors[gltf_primitive.indices > -1 ? gltf_primitive.indices : 0];
            const tinygltf::BufferView &indices_buffer_view =
                gltf_model.bufferViews[indices_accessor.bufferView];
            const tinygltf::Buffer &indices_buffer =
                gltf_model.buffers[indices_buffer_view.buffer];
            const void *indices_data_ptr =
                &(indices_buffer
                      .data[indices_accessor.byteOffset + indices_buffer_view.byteOffset]
                );

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

            // Create geometry and scene for instancing later

            RTCGeometry geom =
                rtcNewGeometry(app.embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);

            rtcSetSharedGeometryBuffer(
                geom,
                RTC_BUFFER_TYPE_VERTEX,
                0,
                RTC_FORMAT_FLOAT3,
                primitive.positions,
                0,
                sizeof(glm::vec3),
                primitive.vertex_count
            );

            assert(primitive.index_count % 3 == 0);
            uint32_t triangle_count = primitive.index_count / 3;
            rtcSetSharedGeometryBuffer(
                geom,
                RTC_BUFFER_TYPE_INDEX,
                0,
                RTC_FORMAT_UINT3,
                primitive.indices,
                0,
                3 * sizeof(uint32_t),
                triangle_count
            );

            rtcCommitGeometry(geom);

            primitive.scene = rtcNewScene(app.embree_device);
            rtcAttachGeometry(primitive.scene, geom);
            rtcCommitScene(primitive.scene);

            rtcReleaseGeometry(geom);
        }
    }
}

void Scene::load_node(
    App &app,
    const tinygltf::Model &gltf_model,
    uint32_t node_index,
    std::optional<uint32_t> parent_index
) {
    const tinygltf::Node &gltf_node = gltf_model.nodes[node_index];
    Node &node = this->nodes[node_index];
    node.parent = parent_index;

    if (gltf_node.camera != -1) {
        this->camera_node_index = node_index;
    }

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
            *geom = rtcNewGeometry(app.embree_device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryTimeStepCount(*geom, 1);
            rtcSetGeometryInstancedScene(*geom, prim.scene);
            glm::mat4 global_transform = this->node_global_matrix(node);
            rtcSetGeometryTransform(
                *geom, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &global_transform[0][0]
            );

            GeometryData *user_data =
                alignedSYCLMallocDeviceReadOnly<GeometryData>(app.queue, 1, 16);
            *user_data = GeometryData{
                .vertex_buffer = prim.positions,
                .normal_buffer = prim.normals,
                .uv_buffer = prim.uvs,
                .index_buffer = prim.indices,
                .obj_to_world = glm::transpose(glm::inverse(glm::mat3(global_transform))),
                .material = prim.material,
            };
            rtcSetGeometryUserData(*geom, user_data);

            rtcCommitGeometry(*geom);
        }
    }
}

} // namespace raytracer
