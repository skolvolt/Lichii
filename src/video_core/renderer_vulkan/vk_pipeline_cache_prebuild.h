// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/unique_function.h"
#include "video_core/renderer_vulkan/pipeline_build.h"

namespace Vulkan {

template <typename Key, typename Pipeline>
class PipelinePrebuild {
    struct Entry {
        std::unique_ptr<EmissionBudget::Reservation> reservation;
        std::unique_ptr<Pipeline> pipeline;
        bool claimed{};
    };

public:
    using Build = Common::UniqueFunction<std::unique_ptr<Pipeline>>;

    PipelinePrebuild(size_t capacity, size_t capture_limit_, bool broken_parallel_compilation)
        : budget{broken_parallel_compilation ? 0 : capacity}, capture_limit{capture_limit_} {}

    ~PipelinePrebuild() {
        Stop();
    }

    /// nup: `reserve` = false (follow-on builds, which read their record when they run) takes no
    /// emission slot, and an Add after Start queues the build at once.
    bool Add(const Key& key, size_t capture_bytes, Build build, bool reserve = true) {
        std::unique_ptr<EmissionBudget::Reservation> reservation;
        {
            std::scoped_lock lock{mutex};
            if (stopped || entries.contains(key) ||
                capture_bytes > capture_limit - captured_bytes) {
                return false;
            }
            if (reserve && !(reservation = budget.TryAcquire(true))) {
                return false;
            }
        }
        auto entry = std::make_shared<Entry>();
        entry->reservation = std::move(reservation);
        captured_bytes += capture_bytes;
        Common::UniqueFunction<void> task([this, key, entry, build = std::move(build)]() mutable {
            bool run;
            {
                std::scoped_lock lock{mutex};
                run = !stopped && !entry->claimed;
            }
            std::unique_ptr<Pipeline> pipeline;
            if (run) {
                try {
                    pipeline = build();
                } catch (const std::exception& exception) {
                    LOG_ERROR(Render_Vulkan, "Cached pipeline prebuild failed: {}",
                              exception.what());
                } catch (...) {
                    LOG_ERROR(Render_Vulkan, "Cached pipeline prebuild failed");
                }
            }
            build = {};
            {
                std::scoped_lock lock{mutex};
                if (!stopped && !entry->claimed && pipeline) {
                    entry->pipeline = std::move(pipeline);
                } else if (!entry->claimed) {
                    entries.erase(key);
                }
            }
        });
        std::scoped_lock lock{mutex};
        entries.emplace(key, entry);
        if (started) {
            workers->QueueWork(std::move(task));
        } else {
            tasks.push_back(std::move(task));
        }
        return true;
    }

    void Start(size_t threads) {
        // nup: starts without tasks too: follow-on builds are added while playing.
        std::scoped_lock lock{mutex};
        if (started || stopped || threads == 0) {
            return;
        }
        started = true;
        workers.reset(new Common::ThreadWorker(threads, "VkPipelinePrebuild", {},
                                               Common::ThreadPlacement::Background));
        for (auto& task : tasks) {
            workers->QueueWork(std::move(task));
        }
        tasks.clear();
    }

    std::unique_ptr<Pipeline> Take(const Key& key) {
        std::shared_ptr<Entry> entry;
        {
            std::scoped_lock lock{mutex};
            const auto it = entries.find(key);
            if (it == entries.end()) {
                return nullptr;
            }
            entry = std::move(it->second);
            entries.erase(it);
            entry->claimed = true;
        }
        return std::move(entry->pipeline);
    }

    void Stop() {
        {
            std::scoped_lock lock{mutex};
            stopped = true;
        }
        workers.reset();
        tasks.clear();
        entries.clear();
    }

private:
    EmissionBudget budget;
    const size_t capture_limit;
    size_t captured_bytes{};
    bool started{};
    bool stopped{};
    std::mutex mutex;
    std::unordered_map<Key, std::shared_ptr<Entry>> entries;
    std::vector<Common::UniqueFunction<void>> tasks;
    std::unique_ptr<Common::ThreadWorker> workers;
};

} // namespace Vulkan
