#pragma once

#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "tiny_gltf.h"

#include "app.hpp"

namespace raytracer {

struct Primitive {
    glm::vec3 *positions;
    size_t position_count;
    uint32_t *indices;
    uint32_t index_count;

    RTCScene scene;
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
