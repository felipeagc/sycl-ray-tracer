#pragma once

#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "tiny_gltf.h"

#include "app.hpp"
#include "xorshift.hpp"
#include "util.hpp"

namespace raytracer {

enum class MaterialType : uint8_t {
    eDiffuse,
    eDielectric,
};

struct MaterialDiffuse {
    sycl::float4 albedo;
};

struct MaterialDielectric {};

struct Material {
    MaterialType type;
    union {
        MaterialDiffuse diffuse;
        MaterialDielectric dielectric;
    };

    Material(MaterialDiffuse diffuse) {
        type = MaterialType::eDiffuse;
        this->diffuse = diffuse;
    }

    Material(MaterialDielectric dielectric) {
        type = MaterialType::eDielectric;
        this->dielectric = dielectric;
    }

    bool scatter(
        const RTCRayHit &rayhit,
        sycl::float4 &attenuation,
        RTCRay &scattered,
        XorShift32State &rng
    ) const {
        scattered.org_x = rayhit.ray.org_x + rayhit.ray.dir_x * rayhit.ray.tfar;
        scattered.org_y = rayhit.ray.org_y + rayhit.ray.dir_y * rayhit.ray.tfar;
        scattered.org_z = rayhit.ray.org_z + rayhit.ray.dir_z * rayhit.ray.tfar;

        scattered.tnear = 0.0001f;
        scattered.tfar = std::numeric_limits<float>::infinity();

        sycl::float3 new_dir;

        sycl::float3 normal =
            normalize(sycl::float3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));

        switch (this->type) {
        case MaterialType::eDiffuse: {
            new_dir = normal + rng.random_unit_vector();
            if (near_zero(new_dir)) {
                new_dir = normal;
            }

            attenuation = this->diffuse.albedo;

            scattered.dir_x = new_dir.x();
            scattered.dir_y = new_dir.y();
            scattered.dir_z = new_dir.z();
            return true;
        }
        case MaterialType::eDielectric: {
            return false;
        }
        }
    }
};

struct GeometryData {
    sycl::float4 emissive;
    Material material = MaterialDiffuse{.albedo = sycl::float4(1, 1, 1, 1)};
};

struct Primitive {
    glm::vec3 *positions;
    size_t position_count;
    uint32_t *indices;
    uint32_t index_count;

    RTCScene scene;
    GeometryData user_data;
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

    ~Node();

    glm::mat4 local_matrix() const;
};

struct Scene {
    std::vector<Node> nodes;
    std::vector<Mesh> meshes;
    glm::vec3 global_scale;
    RTCScene scene;
    int camera_node_index = -1;

    glm::vec3 camera_position;
    glm::vec3 camera_direction;
    float camera_focal_length;

    Scene(const Scene &) = delete;
    Scene &operator=(const Scene &) = delete;

    Scene(Scene &&other);
    Scene &operator=(Scene &&other);

    Scene(
        App &app, const std::string &filepath, glm::vec3 global_scale = {1.0f, 1.0f, 1.0f}
    );

    ~Scene();

    glm::mat4 node_global_matrix(const Node &node) const;

    void load_primitives(App &app, const tinygltf::Model &gltf_model);

    void load_node(
        App &app,
        const tinygltf::Model &gltf_model,
        uint32_t node_index,
        std::optional<uint32_t> parent_index
    );
};

} // namespace raytracer
