#include "llama-chat-tree.h"

#include "llama-context.h"

#include <algorithm>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <cstring>

llama_chat_tree::llama_chat_tree(llama_context * ctx) : ctx_(ctx) {}

int32_t llama_chat_tree::now_s() {
    return static_cast<int32_t>(std::time(nullptr));
}

int32_t llama_chat_tree::clamp_size(size_t value) {
    return value > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        ? std::numeric_limits<int32_t>::max()
        : static_cast<int32_t>(value);
}

const llama_chat_tree_node * llama_chat_tree::find_node(int32_t node_id) const {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return nullptr;
    }
    return &it->second;
}

llama_chat_tree_node & llama_chat_tree::require_node(int32_t node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        throw std::runtime_error("Tree node not found: " + std::to_string(node_id));
    }
    return it->second;
}

void llama_chat_tree::reset_state() {
    nodes_.clear();

    llama_chat_tree_node root;
    root.id = 0;
    root.parent_id = -1;
    root.status = "cached";
    root.prefix_token_count = 0;
    root.generation_time_ms = 0;
    root.cached_token_count = 0;
    root.snapshot_token_bytes = 0;
    root.created_at_s = now_s();
    root.last_accessed_at_s = root.created_at_s;

    nodes_[0] = root;
    root_id_ = 0;
    active_node_id_ = 0;
    next_id_ = 1;
    total_snapshot_token_bytes_ = 0;
    last_pruned_node_ids_.clear();
    last_pruned_at_s_ = 0;
    initialized_ = true;
}

void llama_chat_tree::refresh_context_memory() {
    if (ctx_ == nullptr) {
        context_memory_bytes_ = 0;
        return;
    }

    size_t total = 0;
    for (const auto & entry : ctx_->memory_breakdown()) {
        total += entry.second.context;
    }
    context_memory_bytes_ = clamp_size(total);
}

void llama_chat_tree::refresh_totals() {
    int64_t total = 0;
    for (auto & entry : nodes_) {
        auto & node = entry.second;
        if (node.id == root_id_) {
            node.snapshot_token_bytes = 0;
            continue;
        }
        total += node.snapshot_token_bytes;
    }

    total_snapshot_token_bytes_ = total > std::numeric_limits<int32_t>::max()
        ? std::numeric_limits<int32_t>::max()
        : static_cast<int32_t>(total);

    refresh_context_memory();
}

std::vector<int32_t> llama_chat_tree::path_ids(int32_t node_id) {
    std::vector<int32_t> path;
    int32_t current = node_id;
    while (true) {
        auto & node = require_node(current);
        path.push_back(current);
        if (node.parent_id < 0) {
            break;
        }
        current = node.parent_id;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

void llama_chat_tree::touch_path(int32_t node_id) {
    const int32_t t = now_s();
    for (int32_t path_id : path_ids(node_id)) {
        require_node(path_id).last_accessed_at_s = t;
    }
}

int32_t llama_chat_tree::subtree_max_token_count(int32_t node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return 0;
    }

    auto & node = it->second;
    int32_t result = std::max(0, node.prefix_token_count);
    for (int32_t child_id : node.child_ids) {
        result = std::max(result, subtree_max_token_count(child_id));
    }
    return result;
}

void llama_chat_tree::update_ancestor_cache_count(int32_t node_id) {
    for (int32_t path_id : path_ids(node_id)) {
        require_node(path_id).cached_token_count = subtree_max_token_count(path_id);
    }
}

bool llama_chat_tree::delete_leaf_node(int32_t node_id, std::string & err) {
    auto & node = require_node(node_id);
    if (!node.child_ids.empty()) {
        err = "Tree prune target is not a leaf: " + std::to_string(node_id);
        return false;
    }

    const int32_t parent_id = node.parent_id;
    nodes_.erase(node_id);

    if (parent_id >= 0) {
        auto & parent = require_node(parent_id);
        parent.child_ids.erase(
            std::remove(parent.child_ids.begin(), parent.child_ids.end(), node_id),
            parent.child_ids.end());
        update_ancestor_cache_count(parent_id);
    }

    return true;
}

void llama_chat_tree::prune_to_memory_cap(std::vector<int32_t> & pruned_node_ids) {
    while (total_snapshot_token_bytes_ > memory_cap_bytes_) {
        std::vector<llama_chat_tree_node *> leaf_candidates;
        for (auto & entry : nodes_) {
            auto & node = entry.second;
            if (node.id == root_id_ || node.id == active_node_id_) {
                continue;
            }
            if (node.child_ids.empty()) {
                leaf_candidates.push_back(&node);
            }
        }

        if (leaf_candidates.empty()) {
            break;
        }

        std::sort(
            leaf_candidates.begin(),
            leaf_candidates.end(),
            [](const llama_chat_tree_node * left, const llama_chat_tree_node * right) {
                return left->last_accessed_at_s < right->last_accessed_at_s;
            });

        const int32_t victim_id = leaf_candidates.front()->id;
        std::string err;
        if (!delete_leaf_node(victim_id, err)) {
            break;
        }
        pruned_node_ids.push_back(victim_id);
        refresh_totals();
    }
}

void llama_chat_tree::init(int32_t memory_cap_bytes) {
    memory_cap_bytes_ = memory_cap_bytes;
    reset_state();
    refresh_totals();
}

void llama_chat_tree::reset() {
    reset_state();
    refresh_totals();
}

void llama_chat_tree::ensure_initialized_for_chat() {
    if (!initialized_) {
        // Keep chat APIs self-bootstrapping so upper layers don't need
        // to sequence explicit tree init calls.
        init(memory_cap_bytes_);
    }
}

bool llama_chat_tree::chat_set_active(int32_t node_id, std::string & err) {
    ensure_initialized_for_chat();
    return switch_to(node_id, err);
}

bool llama_chat_tree::chat_delete(int32_t node_id, std::vector<int32_t> & deleted_node_ids, std::string & err) {
    ensure_initialized_for_chat();
    return delete_node(node_id, deleted_node_ids, err);
}

bool llama_chat_tree::switch_to(int32_t node_id, std::string & err) {
    if (!initialized_) {
        err = "Tree is not initialized";
        return false;
    }

    if (node_id == root_id_) {
        active_node_id_ = root_id_;
        touch_path(root_id_);
        return true;
    }

    auto & node = require_node(node_id);
    if (node.status != "cached" || node.prefix_token_count < 0) {
        err = "Tree node is not cached: " + std::to_string(node_id);
        return false;
    }

    active_node_id_ = node_id;
    touch_path(node_id);
    return true;
}

bool llama_chat_tree::prepare_turn(int32_t parent_id, const std::string & user_text, int32_t & node_id, std::string & err) {
    if (!initialized_) {
        err = "Tree is not initialized";
        return false;
    }

    require_node(parent_id);

    const int32_t t = now_s();
    node_id = next_id_++;

    llama_chat_tree_node node;
    node.id = node_id;
    node.parent_id = parent_id;
    node.user_text = user_text;
    node.status = "generating";
    node.created_at_s = t;
    node.last_accessed_at_s = t;

    nodes_[node_id] = std::move(node);
    require_node(parent_id).child_ids.push_back(node_id);
    active_node_id_ = node_id;
    touch_path(node_id);

    return true;
}

bool llama_chat_tree::finish_turn(
    int32_t node_id,
    const std::string & assistant_text,
    int32_t generation_time_ms,
    int32_t prefix_token_count,
    int32_t snapshot_token_bytes,
    std::vector<int32_t> & pruned_node_ids,
    std::string & err) {

    if (!initialized_) {
        err = "Tree is not initialized";
        return false;
    }

    auto & node = require_node(node_id);
    node.assistant_text = assistant_text;
    node.generation_time_ms = generation_time_ms;
    node.prefix_token_count = prefix_token_count;
    node.cached_token_count = prefix_token_count;
    node.snapshot_token_bytes = std::max(0, snapshot_token_bytes);
    node.status = "cached";

    touch_path(node.id);
    update_ancestor_cache_count(node.id);
    refresh_totals();

    pruned_node_ids.clear();
    prune_to_memory_cap(pruned_node_ids);

    last_pruned_node_ids_ = pruned_node_ids;
    if (!pruned_node_ids.empty()) {
        last_pruned_at_s_ = now_s();
    }

    return true;
}

bool llama_chat_tree::delete_node(int32_t node_id, std::vector<int32_t> & deleted_node_ids, std::string & err) {
    if (!initialized_) {
        err = "Tree is not initialized";
        return false;
    }

    if (node_id == root_id_) {
        err = "Cannot delete root node";
        return false;
    }

    auto & node = require_node(node_id);
    const int32_t parent_id = node.parent_id;

    std::vector<int32_t> queue = { node_id };
    deleted_node_ids.clear();

    for (size_t i = 0; i < queue.size(); ++i) {
        const int32_t current_id = queue[i];
        deleted_node_ids.push_back(current_id);
        auto & current = require_node(current_id);
        queue.insert(queue.end(), current.child_ids.begin(), current.child_ids.end());
    }

    if (std::find(deleted_node_ids.begin(), deleted_node_ids.end(), active_node_id_) != deleted_node_ids.end()) {
        active_node_id_ = parent_id >= 0 ? parent_id : root_id_;
    }

    for (int32_t id : deleted_node_ids) {
        nodes_.erase(id);
    }

    if (parent_id >= 0 && nodes_.find(parent_id) != nodes_.end()) {
        auto & parent = require_node(parent_id);
        parent.child_ids.erase(
            std::remove(parent.child_ids.begin(), parent.child_ids.end(), node_id),
            parent.child_ids.end());
        update_ancestor_cache_count(parent_id);
    }

    refresh_totals();
    return true;
}

bool llama_chat_tree::chat_start(int32_t parent_id, const std::string & user_text, int32_t & node_id, std::string & err) {
    ensure_initialized_for_chat();

    if (!find_node(parent_id)) {
        err = "Tree node not found: " + std::to_string(parent_id);
        return false;
    }

    // Track that we are branching from this parent. KV restore is handled by bridge layer.
    if (!switch_to(parent_id, err)) {
        return false;
    }

    return prepare_turn(parent_id, user_text, node_id, err);
}

bool llama_chat_tree::chat_finish(
    int32_t node_id,
    const std::string & assistant_text,
    int32_t generation_time_ms,
    int32_t prefix_token_count,
    int32_t snapshot_token_bytes,
    bool aborted_or_error,
    std::vector<int32_t> & pruned_node_ids,
    std::vector<int32_t> & deleted_node_ids,
    std::string & err) {

    ensure_initialized_for_chat();

    pruned_node_ids.clear();
    deleted_node_ids.clear();

    if (aborted_or_error) {
        return delete_node(node_id, deleted_node_ids, err);
    }

    return finish_turn(
        node_id,
        assistant_text,
        generation_time_ms,
        prefix_token_count,
        snapshot_token_bytes,
        pruned_node_ids,
        err);
}

bool llama_chat_tree::collect_chat_messages(
    int32_t node_id,
    std::vector<std::string> & roles,
    std::vector<std::string> & contents,
    std::string & err) const {

    roles.clear();
    contents.clear();

    const llama_chat_tree_node * cur = find_node(node_id);
    if (cur == nullptr) {
        err = "Tree node not found: " + std::to_string(node_id);
        return false;
    }

    std::vector<const llama_chat_tree_node *> path;
    while (cur != nullptr) {
        path.push_back(cur);
        if (cur->parent_id < 0) {
            break;
        }
        cur = find_node(cur->parent_id);
    }
    std::reverse(path.begin(), path.end());

    for (const auto * node : path) {
        if (node == nullptr || node->id == root_id_) {
            continue;
        }
        if (!node->user_text.empty()) {
            roles.push_back("user");
            contents.push_back(node->user_text);
        }
        if (!node->assistant_text.empty()) {
            roles.push_back("assistant");
            contents.push_back(node->assistant_text);
        }
    }

    return true;
}

bool llama_chat_tree::chat_format_prompt(
    int32_t node_id,
    const std::string & tmpl,
    bool add_ass,
    std::string & formatted_prompt,
    std::string & err) const {

    std::vector<std::string> roles;
    std::vector<std::string> contents;
    if (!collect_chat_messages(node_id, roles, contents, err)) {
        return false;
    }

    std::vector<llama_chat_message> chat;
    chat.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        chat.push_back({ roles[i].c_str(), contents[i].c_str() });
    }

    const llama_model * model = ctx_ ? llama_get_model(ctx_) : nullptr;
    const char * model_tmpl = model ? llama_model_chat_template(model, nullptr) : nullptr;
    const char * ptr_tmpl = tmpl.empty() ? model_tmpl : tmpl.c_str();

    int alloc_size = 0;
    for (size_t i = 0; i < roles.size(); ++i) {
        alloc_size += (int)((roles[i].size() + contents[i].size()) * 1.25);
    }
    alloc_size = std::max(alloc_size, 256);

    std::vector<char> buf((size_t)alloc_size);
    int32_t res = llama_chat_apply_template(ptr_tmpl, chat.data(), chat.size(), add_ass, buf.data(), (int32_t)buf.size());

    bool fallback_chatml = false;
    if (res < 0) {
        if (ptr_tmpl != nullptr) {
            err = "custom chat template is not supported";
            return false;
        }
        res = llama_chat_apply_template("chatml", chat.data(), chat.size(), add_ass, buf.data(), (int32_t)buf.size());
        fallback_chatml = true;
        if (res < 0) {
            err = "failed to apply chat template";
            return false;
        }
    }

    if ((size_t)res > buf.size()) {
        buf.resize((size_t)res + 1);
        const char * retry_tmpl = fallback_chatml ? "chatml" : ptr_tmpl;
        res = llama_chat_apply_template(retry_tmpl, chat.data(), chat.size(), add_ass, buf.data(), (int32_t)buf.size());
        if (res < 0) {
            err = "failed to apply chat template after resize";
            return false;
        }
    }

    if (res >= (int32_t)buf.size()) {
        buf.resize((size_t)res + 1);
    }
    buf[(size_t)res] = '\0';
    formatted_prompt.assign(buf.data(), (size_t)res);
    return true;
}
