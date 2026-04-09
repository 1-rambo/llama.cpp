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

enum llama_chat_tree_cache_tier : int32_t {
    LLAMA_CHAT_TREE_CACHE_TIER_L1 = 1,
    LLAMA_CHAT_TREE_CACHE_TIER_L2 = 2,
    LLAMA_CHAT_TREE_CACHE_TIER_L3 = 3,
};

enum llama_chat_tree_replacement_policy : int32_t {
    LLAMA_CHAT_TREE_REPLACEMENT_HYBRID = 0,
    LLAMA_CHAT_TREE_REPLACEMENT_LRU = 1,
    LLAMA_CHAT_TREE_REPLACEMENT_LFU = 2,
    LLAMA_CHAT_TREE_REPLACEMENT_SIZE_ONLY = 3,
    LLAMA_CHAT_TREE_REPLACEMENT_RANDOM = 4,
};

struct llama_chat_tree_tier_config {
    bool enabled = false;
    llama_chat_tree_replacement_policy replacement_policy = LLAMA_CHAT_TREE_REPLACEMENT_HYBRID;
    int32_t l1_token_cap = 0;
    int32_t l2_token_cap = 0;
    int32_t l3_token_cap = 0;
    // If > 0 and node token count >= threshold, prefer pruning over demotion
    // when deciding L1 -> L2 boundary handling.
    int32_t prune_l1_l2_token_threshold = 0;
    // If > 0 and node token count >= threshold, prefer pruning over demotion
    // when deciding L2 -> L3 boundary handling.
    int32_t prune_l2_l3_token_threshold = 0;
    std::string l3_path = "/tmp/wllama-tier-cache";
};

struct llama_chat_tree_tier_stats {
    int32_t promotions = 0;
    int32_t demotions = 0;
    int32_t disk_reads = 0;
    int32_t disk_writes = 0;
    int32_t l3_overflow_events = 0;
    int32_t restore_attempts = 0;
    int32_t restore_hits_l1 = 0;
    int32_t restore_hits_l2 = 0;
    int32_t restore_hits_l3 = 0;
    int32_t restore_misses = 0;
    int32_t restore_rebuilds = 0;
    int32_t parent_recover_attempts = 0;
    int32_t parent_recover_successes = 0;
    int32_t parent_recover_failures = 0;
    int32_t slot_alloc_hits = 0;
    int32_t slot_alloc_misses = 0;
    int32_t slot_evict_l1 = 0;
    int32_t slot_evict_l2 = 0;
    int32_t slot_evict_l3 = 0;
    int32_t fallback_replays = 0;
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
    bool chat_recover_parent(int32_t node_id, std::string & err);
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

    void set_tier_config(const llama_chat_tree_tier_config & config);
    const llama_chat_tree_tier_config & tier_config() const { return tier_config_; }
    const llama_chat_tree_tier_stats & tier_stats() const { return tier_stats_; }
    void tier_reset();
    void tier_on_slot_saved(int32_t slot_id, int32_t token_count);
    void tier_on_slot_removed(int32_t slot_id);
    void tier_on_slot_restored(int32_t slot_id, int32_t token_count);
    void tier_on_disk_read();
    void tier_on_disk_write();
    void tier_on_l3_overflow();
    void tier_on_restore_attempt();
    void tier_on_restore_hit(int32_t level);
    void tier_on_restore_miss();
    void tier_on_restore_rebuild();
    void tier_on_slot_alloc_hit();
    void tier_on_slot_alloc_miss();
    void tier_on_slot_evict(int32_t level);
    void tier_on_fallback_replay();
    int32_t tier_slot_level(int32_t slot_id) const;
    bool tier_set_slot_level(int32_t slot_id, int32_t level);
    int32_t tier_slot_token_count(int32_t slot_id) const;
    int32_t tier_total_tokens(int32_t level) const;
    int32_t tier_total_slots(int32_t level) const;
    // Pick a replacement victim by adaptive score (age/size/frequency), not strict LRU.
    int32_t tier_pick_victim_slot(int32_t level, int32_t excluded_slot) const;
    bool should_prune_on_l1_l2_boundary(const llama_chat_tree_node & node) const;
    bool should_prune_on_l2_l3_boundary(const llama_chat_tree_node & node) const;
    bool should_force_prune_l3_over_cap(const llama_chat_tree_node & node) const;

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

    struct tier_slot_meta {
        int32_t token_count = 0;
        int32_t level = LLAMA_CHAT_TREE_CACHE_TIER_L1;
        int64_t last_access_tick = 0;
        int32_t access_count = 0;
    };

    std::unordered_map<int32_t, tier_slot_meta> tier_slots_;

    struct archived_node_meta {
        int32_t id = 0;
        int32_t parent_id = -1;
        std::string user_text;
        std::string assistant_text;
        std::string status;
        int32_t prefix_token_count = -1;
        int32_t generation_time_ms = -1;
        int32_t cached_token_count = 0;
        int32_t snapshot_token_bytes = 0;
    };

    std::unordered_map<int32_t, archived_node_meta> archived_nodes_;
    int64_t tier_access_tick_ = 0;
    llama_chat_tree_tier_config tier_config_;
    llama_chat_tree_tier_stats tier_stats_;

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
    void archive_node(const llama_chat_tree_node & node);
    bool recover_parent_chain(int32_t node_id, std::string & err);
    bool should_prune_leaf(const llama_chat_tree_node & node) const;
    void prune_to_memory_cap(std::vector<int32_t> & pruned_node_ids);
};
