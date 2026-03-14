#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_chat_tree_node {
    int32_t id = 0;
    int32_t parent_id = -1;
    std::vector<int32_t> child_ids;
    std::string user_text;
    std::string assistant_text;
    std::string status = "pending";
    int32_t prefix_token_count = -1;
    int32_t generation_time_ms = -1;
    int32_t cached_token_count = 0;
    int32_t snapshot_token_bytes = 0;
    int32_t created_at_s = 0;
    int32_t last_accessed_at_s = 0;
};

class llama_chat_tree {
public:
    explicit llama_chat_tree(llama_context * ctx);

    void init(int32_t memory_cap_bytes);
    void reset();

    // Application-level transaction APIs used by chat-completions flow.
    bool chat_set_active(int32_t node_id, std::string & err);
    bool chat_delete(int32_t node_id, std::vector<int32_t> & deleted_node_ids, std::string & err);
    bool chat_start(int32_t parent_id, const std::string & user_text, int32_t & node_id, std::string & err);
    bool chat_finish(
        int32_t node_id,
        const std::string & assistant_text,
        int32_t generation_time_ms,
        int32_t prefix_token_count,
        int32_t snapshot_token_bytes,
        bool aborted_or_error,
        std::vector<int32_t> & pruned_node_ids,
        std::vector<int32_t> & deleted_node_ids,
        std::string & err);

    bool chat_format_prompt(
        int32_t node_id,
        const std::string & tmpl,
        bool add_ass,
        std::string & formatted_prompt,
        std::string & err) const;

    bool collect_chat_messages(int32_t node_id, std::vector<std::string> & roles, std::vector<std::string> & contents, std::string & err) const;

    bool initialized() const { return initialized_; }

    const std::unordered_map<int32_t, llama_chat_tree_node> & nodes() const { return nodes_; }
    const llama_chat_tree_node * find_node(int32_t node_id) const;

    int32_t root_id() const { return root_id_; }
    int32_t active_node_id() const { return active_node_id_; }
    int32_t next_id() const { return next_id_; }
    int32_t context_memory_bytes() const { return context_memory_bytes_; }
    int32_t memory_cap_bytes() const { return memory_cap_bytes_; }
    int32_t total_snapshot_token_bytes() const { return total_snapshot_token_bytes_; }
    const std::vector<int32_t> & last_pruned_node_ids() const { return last_pruned_node_ids_; }
    int32_t last_pruned_at_s() const { return last_pruned_at_s_; }

private:
    llama_context * ctx_ = nullptr;

    std::unordered_map<int32_t, llama_chat_tree_node> nodes_;
    int32_t root_id_ = 0;
    int32_t active_node_id_ = 0;
    int32_t next_id_ = 1;
    int32_t context_memory_bytes_ = 0;
    int32_t memory_cap_bytes_ = 1024 * 1024 * 1024;
    int32_t total_snapshot_token_bytes_ = 0;
    std::vector<int32_t> last_pruned_node_ids_;
    int32_t last_pruned_at_s_ = 0;
    bool initialized_ = false;

    static int32_t now_s();
    static int32_t clamp_size(size_t value);
    void ensure_initialized_for_chat();

    // Low-level tree mutators are intentionally private to keep the public
    // surface chat-transaction-oriented.
    bool switch_to(int32_t node_id, std::string & err);
    bool prepare_turn(int32_t parent_id, const std::string & user_text, int32_t & node_id, std::string & err);
    bool finish_turn(
        int32_t node_id,
        const std::string & assistant_text,
        int32_t generation_time_ms,
        int32_t prefix_token_count,
        int32_t snapshot_token_bytes,
        std::vector<int32_t> & pruned_node_ids,
        std::string & err);
    bool delete_node(int32_t node_id, std::vector<int32_t> & deleted_node_ids, std::string & err);

    llama_chat_tree_node & require_node(int32_t node_id);

    void reset_state();
    void touch_path(int32_t node_id);
    std::vector<int32_t> path_ids(int32_t node_id);

    int32_t subtree_max_token_count(int32_t node_id);
    void update_ancestor_cache_count(int32_t node_id);
    void refresh_totals();
    void refresh_context_memory();

    bool delete_leaf_node(int32_t node_id, std::string & err);
    void prune_to_memory_cap(std::vector<int32_t> & pruned_node_ids);
};
