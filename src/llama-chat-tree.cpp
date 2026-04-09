#include "llama-chat-tree.h"

#include "llama-context.h"

#include <algorithm>
#include <cmath>
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
    archived_nodes_.clear();

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

void llama_chat_tree::set_tier_config(const llama_chat_tree_tier_config & config) {
    tier_config_ = config;
}

void llama_chat_tree::tier_reset() {
    tier_slots_.clear();
    tier_access_tick_ = 0;
    tier_stats_ = llama_chat_tree_tier_stats{};
}

void llama_chat_tree::tier_on_slot_saved(int32_t slot_id, int32_t token_count) {
    if (slot_id < 1) {
        return;
    }

    tier_slot_meta meta;
    auto it = tier_slots_.find(slot_id);
    if (it != tier_slots_.end()) {
        meta = it->second;
    }

    tier_access_tick_++;
    meta.token_count = std::max(0, token_count);
    meta.level = LLAMA_CHAT_TREE_CACHE_TIER_L1;
    meta.last_access_tick = tier_access_tick_;
    meta.access_count = std::max(1, meta.access_count + 1);
    tier_slots_[slot_id] = meta;
}

void llama_chat_tree::tier_on_slot_removed(int32_t slot_id) {
    tier_slots_.erase(slot_id);
}

void llama_chat_tree::tier_on_slot_restored(int32_t slot_id, int32_t token_count) {
    if (slot_id < 1) {
        return;
    }

    auto & meta = tier_slots_[slot_id];
    tier_access_tick_++;
    meta.token_count = std::max(0, token_count);
    if (meta.level != LLAMA_CHAT_TREE_CACHE_TIER_L1) {
        tier_stats_.promotions++;
    }
    meta.level = LLAMA_CHAT_TREE_CACHE_TIER_L1;
    meta.last_access_tick = tier_access_tick_;
    meta.access_count = std::max(1, meta.access_count + 1);
}

void llama_chat_tree::tier_on_disk_read() {
    tier_stats_.disk_reads++;
}

void llama_chat_tree::tier_on_disk_write() {
    tier_stats_.disk_writes++;
}

void llama_chat_tree::tier_on_l3_overflow() {
    tier_stats_.l3_overflow_events++;
}

void llama_chat_tree::tier_on_restore_attempt() {
    tier_stats_.restore_attempts++;
}

void llama_chat_tree::tier_on_restore_hit(int32_t level) {
    if (level == LLAMA_CHAT_TREE_CACHE_TIER_L1) {
        tier_stats_.restore_hits_l1++;
        return;
    }
    if (level == LLAMA_CHAT_TREE_CACHE_TIER_L2) {
        tier_stats_.restore_hits_l2++;
        return;
    }
    if (level == LLAMA_CHAT_TREE_CACHE_TIER_L3) {
        tier_stats_.restore_hits_l3++;
    }
}

void llama_chat_tree::tier_on_restore_miss() {
    tier_stats_.restore_misses++;
}

void llama_chat_tree::tier_on_restore_rebuild() {
    tier_stats_.restore_rebuilds++;
}

void llama_chat_tree::tier_on_slot_alloc_hit() {
    tier_stats_.slot_alloc_hits++;
}

void llama_chat_tree::tier_on_slot_alloc_miss() {
    tier_stats_.slot_alloc_misses++;
}

void llama_chat_tree::tier_on_slot_evict(int32_t level) {
    if (level == LLAMA_CHAT_TREE_CACHE_TIER_L1) {
        tier_stats_.slot_evict_l1++;
        return;
    }
    if (level == LLAMA_CHAT_TREE_CACHE_TIER_L2) {
        tier_stats_.slot_evict_l2++;
        return;
    }
    if (level == LLAMA_CHAT_TREE_CACHE_TIER_L3) {
        tier_stats_.slot_evict_l3++;
    }
}

void llama_chat_tree::tier_on_fallback_replay() {
    tier_stats_.fallback_replays++;
}

int32_t llama_chat_tree::tier_slot_level(int32_t slot_id) const {
    auto it = tier_slots_.find(slot_id);
    if (it == tier_slots_.end()) {
        return -1;
    }
    return it->second.level;
}

bool llama_chat_tree::tier_set_slot_level(int32_t slot_id, int32_t level) {
    auto it = tier_slots_.find(slot_id);
    if (it == tier_slots_.end()) {
        return false;
    }

    const int32_t old_level = it->second.level;

    if (old_level != level) {
        if (old_level > level) {
            tier_stats_.promotions++;
        } else {
            tier_stats_.demotions++;
        }
    }

    tier_access_tick_++;
    it->second.level = level;
    it->second.last_access_tick = tier_access_tick_;
    if (old_level == level || old_level > level) {
        it->second.access_count = std::max(1, it->second.access_count + 1);
    }
    return true;
}

int32_t llama_chat_tree::tier_slot_token_count(int32_t slot_id) const {
    auto it = tier_slots_.find(slot_id);
    if (it == tier_slots_.end()) {
        return 0;
    }
    return std::max(0, it->second.token_count);
}

int32_t llama_chat_tree::tier_total_tokens(int32_t level) const {
    int64_t total = 0;
    for (const auto & entry : tier_slots_) {
        if (entry.second.level != level) {
            continue;
        }
        total += std::max(0, entry.second.token_count);
    }
    return total > std::numeric_limits<int32_t>::max()
        ? std::numeric_limits<int32_t>::max()
        : static_cast<int32_t>(total);
}

int32_t llama_chat_tree::tier_total_slots(int32_t level) const {
    int32_t total = 0;
    for (const auto & entry : tier_slots_) {
        if (entry.second.level == level) {
            total++;
        }
    }
    return total;
}

int32_t llama_chat_tree::tier_pick_victim_slot(int32_t level, int32_t excluded_slot) const {
    const int64_t now_tick = std::max<int64_t>(1, tier_access_tick_);

    std::vector<int32_t> candidates;
    candidates.reserve(tier_slots_.size());
    for (const auto & entry : tier_slots_) {
        const int32_t slot_id = entry.first;
        if (slot_id == excluded_slot || entry.second.level != level) {
            continue;
        }
        candidates.push_back(slot_id);
    }
    if (candidates.empty()) {
        return -1;
    }

    if (tier_config_.replacement_policy == LLAMA_CHAT_TREE_REPLACEMENT_RANDOM) {
        const uint64_t seed =
            static_cast<uint64_t>(now_tick) * 1103515245ULL +
            static_cast<uint64_t>(std::max(0, level)) * 12345ULL +
            static_cast<uint64_t>(candidates.size()) * 2654435761ULL;
        const size_t idx = static_cast<size_t>(seed % std::max<size_t>(1, candidates.size()));
        return candidates[idx];
    }

    if (tier_config_.replacement_policy == LLAMA_CHAT_TREE_REPLACEMENT_LRU) {
        int32_t best_slot = -1;
        int64_t best_tick = std::numeric_limits<int64_t>::max();
        int32_t best_tokens = -1;
        for (const int32_t slot_id : candidates) {
            const auto & meta = tier_slots_.at(slot_id);
            const int64_t tick = meta.last_access_tick;
            const int32_t tokens = std::max(0, meta.token_count);
            if (tick < best_tick || (tick == best_tick && tokens > best_tokens)) {
                best_tick = tick;
                best_tokens = tokens;
                best_slot = slot_id;
            }
        }
        return best_slot;
    }

    if (tier_config_.replacement_policy == LLAMA_CHAT_TREE_REPLACEMENT_LFU) {
        int32_t best_slot = -1;
        int32_t best_freq = std::numeric_limits<int32_t>::max();
        int64_t best_tick = std::numeric_limits<int64_t>::max();
        int32_t best_tokens = -1;
        for (const int32_t slot_id : candidates) {
            const auto & meta = tier_slots_.at(slot_id);
            const int32_t freq = std::max(1, meta.access_count);
            const int64_t tick = meta.last_access_tick;
            const int32_t tokens = std::max(0, meta.token_count);
            if (freq < best_freq ||
                (freq == best_freq && tick < best_tick) ||
                (freq == best_freq && tick == best_tick && tokens > best_tokens)) {
                best_freq = freq;
                best_tick = tick;
                best_tokens = tokens;
                best_slot = slot_id;
            }
        }
        return best_slot;
    }

    if (tier_config_.replacement_policy == LLAMA_CHAT_TREE_REPLACEMENT_SIZE_ONLY) {
        int32_t best_slot = -1;
        int32_t best_tokens = -1;
        int64_t best_tick = std::numeric_limits<int64_t>::max();
        for (const int32_t slot_id : candidates) {
            const auto & meta = tier_slots_.at(slot_id);
            const int32_t tokens = std::max(0, meta.token_count);
            const int64_t tick = meta.last_access_tick;
            if (tokens > best_tokens || (tokens == best_tokens && tick < best_tick)) {
                best_tokens = tokens;
                best_tick = tick;
                best_slot = slot_id;
            }
        }
        return best_slot;
    }

    // Size normalization uses configured tier cap when available.
    // If cap is unset, fall back to observed max token count in this level.
    const int32_t configured_cap =
        level == LLAMA_CHAT_TREE_CACHE_TIER_L1 ? std::max(0, tier_config_.l1_token_cap) :
        level == LLAMA_CHAT_TREE_CACHE_TIER_L2 ? std::max(0, tier_config_.l2_token_cap) :
        level == LLAMA_CHAT_TREE_CACHE_TIER_L3 ? std::max(0, tier_config_.l3_token_cap) : 0;

    int32_t observed_max_tokens = 1;
    if (configured_cap <= 0) {
        for (const auto & entry : tier_slots_) {
            if (entry.second.level != level) {
                continue;
            }
            observed_max_tokens = std::max(observed_max_tokens, std::max(1, entry.second.token_count));
        }
    }
    const double size_scale = static_cast<double>(configured_cap > 0 ? configured_cap : observed_max_tokens);

    double best_score = -1.0;
    int32_t best_slot = -1;
    for (const int32_t slot_id : candidates) {
        const auto & entry = tier_slots_.at(slot_id);

        const int64_t age = std::max<int64_t>(1, now_tick - entry.last_access_tick);
        const double token_count = static_cast<double>(std::max(1, entry.token_count));
        const double access_count = static_cast<double>(std::max(1, entry.access_count));

        // Monotonic and scale-stable score:
        // - higher age => more evictable (log1p smooths very old outliers)
        // - larger token footprint => more evictable (normalized by tier capacity)
        // - higher access count => less evictable
        const double age_term = std::log1p(static_cast<double>(age));
        const double size_term = token_count / std::max(1.0, size_scale);
        const double freq_term = std::log1p(access_count);
        const double score = (age_term * (1.0 + size_term)) / (1.0 + freq_term);
        if (score > best_score) {
            best_score = score;
            best_slot = slot_id;
            continue;
        }

        if (score == best_score && best_slot >= 0) {
            const auto & best_meta = tier_slots_.at(best_slot);
            if (entry.last_access_tick < best_meta.last_access_tick) {
                best_slot = slot_id;
                continue;
            }
            if (entry.last_access_tick == best_meta.last_access_tick &&
                entry.token_count > best_meta.token_count) {
                best_slot = slot_id;
            }
        }
        if (best_slot < 0) {
            best_slot = slot_id;
        }
    }
    return best_slot;
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
    archive_node(node);
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

void llama_chat_tree::archive_node(const llama_chat_tree_node & node) {
    if (node.id == root_id_) {
        return;
    }
    archived_node_meta meta;
    meta.id = node.id;
    meta.parent_id = node.parent_id;
    meta.user_text = node.user_text;
    meta.assistant_text = node.assistant_text;
    meta.status = node.status;
    meta.prefix_token_count = node.prefix_token_count;
    meta.generation_time_ms = node.generation_time_ms;
    meta.cached_token_count = node.cached_token_count;
    meta.snapshot_token_bytes = node.snapshot_token_bytes;
    archived_nodes_[node.id] = std::move(meta);
}

bool llama_chat_tree::recover_parent_chain(int32_t node_id, std::string & err) {
    if (node_id == root_id_) {
        return true;
    }
    if (find_node(node_id)) {
        return true;
    }

    std::vector<archived_node_meta> chain;
    int32_t cursor = node_id;
    while (find_node(cursor) == nullptr) {
        const auto it = archived_nodes_.find(cursor);
        if (it == archived_nodes_.end()) {
            err = "Tree node not found and no archived lineage: " + std::to_string(cursor);
            return false;
        }
        chain.push_back(it->second);
        if (it->second.parent_id < 0) {
            break;
        }
        cursor = it->second.parent_id;
    }

    if (chain.empty()) {
        err = "Tree recover chain is empty for node: " + std::to_string(node_id);
        return false;
    }

    int32_t current_parent = chain.back().parent_id;
    if (current_parent < 0) {
        current_parent = root_id_;
    }
    if (!find_node(current_parent)) {
        err = "Tree recover anchor not found: " + std::to_string(current_parent);
        return false;
    }

    const int32_t t = now_s();
    for (int i = (int)chain.size() - 1; i >= 0; --i) {
        const archived_node_meta & meta = chain[(size_t)i];
        if (find_node(meta.id)) {
            current_parent = meta.id;
            continue;
        }

        llama_chat_tree_node restored;
        restored.id = meta.id;
        restored.parent_id = current_parent;
        restored.user_text = meta.user_text;
        restored.assistant_text = meta.assistant_text;
        restored.status = "cached";
        restored.prefix_token_count = std::max(0, meta.prefix_token_count);
        restored.generation_time_ms = std::max(0, meta.generation_time_ms);
        restored.cached_token_count = std::max(0, meta.cached_token_count);
        restored.snapshot_token_bytes = std::max(0, meta.snapshot_token_bytes);
        restored.created_at_s = t;
        restored.last_accessed_at_s = t;

        nodes_[restored.id] = restored;
        auto & parent = require_node(current_parent);
        if (std::find(parent.child_ids.begin(), parent.child_ids.end(), restored.id) == parent.child_ids.end()) {
            parent.child_ids.push_back(restored.id);
        }
        if (next_id_ <= restored.id) {
            next_id_ = restored.id + 1;
        }
        current_parent = restored.id;
    }

    touch_path(current_parent);
    update_ancestor_cache_count(current_parent);
    refresh_totals();
    return find_node(node_id) != nullptr;
}

bool llama_chat_tree::should_prune_leaf(const llama_chat_tree_node & node) const {
    if (!tier_config_.enabled) {
        return true;
    }

    // Tiered mode fallback: prefer preserving cache unless L3 over-cap handling
    // or memory-cap pressure forces pruning.
    auto it = tier_slots_.find(node.id);
    if (it == tier_slots_.end()) {
        return true;
    }

    if (it->second.level != LLAMA_CHAT_TREE_CACHE_TIER_L3) {
        return false;
    }

    return should_force_prune_l3_over_cap(node);
}

bool llama_chat_tree::should_prune_on_l1_l2_boundary(const llama_chat_tree_node & node) const {
    if (!tier_config_.enabled) {
        return true;
    }
    const int32_t threshold = std::max(0, tier_config_.prune_l1_l2_token_threshold);
    if (threshold <= 0) {
        return false;
    }
    const int32_t token_count = std::max(0, node.prefix_token_count);
    return token_count >= threshold;
}

bool llama_chat_tree::should_prune_on_l2_l3_boundary(const llama_chat_tree_node & node) const {
    if (!tier_config_.enabled) {
        return true;
    }
    const int32_t threshold = std::max(0, tier_config_.prune_l2_l3_token_threshold);
    if (threshold <= 0) {
        return false;
    }
    const int32_t token_count = std::max(0, node.prefix_token_count);
    return token_count >= threshold;
}

bool llama_chat_tree::should_force_prune_l3_over_cap(const llama_chat_tree_node & node) const {
    if (!tier_config_.enabled) {
        return true;
    }

    // L3 over-cap has no lower tier to demote to; prune is the only option.
    auto it = tier_slots_.find(node.id);
    if (it == tier_slots_.end()) {
        return true;
    }
    if (it->second.level != LLAMA_CHAT_TREE_CACHE_TIER_L3) {
        return false;
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
            if (node.child_ids.empty() && should_prune_leaf(node)) {
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
    tier_reset();
    refresh_totals();
}

void llama_chat_tree::reset() {
    reset_state();
    tier_reset();
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
        auto it = nodes_.find(id);
        if (it != nodes_.end()) {
            archive_node(it->second);
        }
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

bool llama_chat_tree::chat_recover_parent(int32_t node_id, std::string & err) {
    ensure_initialized_for_chat();
    tier_stats_.parent_recover_attempts++;
    const bool ok = recover_parent_chain(node_id, err);
    if (ok) {
        tier_stats_.parent_recover_successes++;
    } else {
        tier_stats_.parent_recover_failures++;
    }
    return ok;
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
