// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/asio.hpp>
#include <future>
#include <iostream>
#include <semaphore>

#include "tt_metal/common/thread_pool.hpp"
#include <tt-metalium/assert.hpp>
namespace tt::tt_metal {

namespace detail {

class BoostThreadPool : public ThreadPool {
public:
    BoostThreadPool(size_t thread_count) : pool_(thread_count) {
        // Given the current use case, we don't expect to
        // enqueue more tasks than the number of threads.
        // Add a factor of safety and modify as needed.
        futures_.reserve(thread_count * 4);
    }

    ~BoostThreadPool() noexcept override = default;

    void enqueue(std::function<void()>&& f) override {
        std::packaged_task<void()> task(std::move(f));
        futures_.push_back(task.get_future());
        boost::asio::post(pool_, [executor = std::move(task)]() mutable { executor(); });
    }

    void wait() override {
        for (auto& future : futures_) {
            future.get();
        }
        futures_.clear();
    }

private:
    boost::asio::thread_pool pool_;
    std::vector<std::future<void>> futures_;
};

class CThreadPool : public ThreadPool {
public:
    CThreadPool(size_t thread_count) {
        workers_.reserve(thread_count);
        application_thread_id_ = std::this_thread::get_id();
        for (size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] {
                // std::cout << "Start Thread" << std::endl;
                while (true) {
                    std::function<void()> task;  // Task container for this thread
                    {
                        task_semaphore_.acquire();  // Ensures 1:1 task to worker mapping
                        if (shutdown_) {
                            return;
                        }
                        // The lock free queue only allows a single reader/single writer
                        // With multiple readers, we must use a lock to synchronize
                        std::unique_lock<std::mutex> lock(mutex_);
                        task = std::move(tasks_.pop());  // Move the function out of the queue
                    }
                    task();  // Execute the function
                    // Atomically decrement counter used to synchronize with main thread
                    // and notify the main thread if all tasks have completed
                    if (counter_.fetch_sub(1, std::memory_order_release) == 1) {
                        counter_.notify_all();
                    }
                }
            });
        }
    }

    ~CThreadPool() noexcept override {
        shutdown_ = true;
        // Notify all workers that a shutdown signal was sent
        for (size_t i = 0; i < workers_.size(); ++i) {
            task_semaphore_.release();
        }
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    void enqueue(std::function<void()>&& f) override {
        TT_ASSERT(
            std::this_thread::get_id() == application_thread_id_,
            "Can only push to thread-pool from the owning thread.");
        tasks_.push(std::move(f));  // Move the task directly into queue
        task_semaphore_.release();  // Notify a worker that a task is available
        // Light-Weight counter increment to track the number of tasks in flight
        // Need this because a counting_semaphore does not allow querying state
        counter_++;
    }
    void wait() override {
        // Wait until all tasks have completed (counter_ == 0)
        // To avoid spinning, sleep until notified by the worker threads
        // or counter_ changes (this only happens with a spurious wakeup)
        int current;
        while ((current = counter_.load(std::memory_order_acquire)) > 0) {
            counter_.wait(current, std::memory_order_relaxed);
        }
    }

private:
    class WorkerQueue {
    public:
        WorkerQueue() {
            // Initialize ring buffer for traversal. Each node points to the subsequent node, except for the last one,
            // which points to the head.
            for (int node_idx = 0; node_idx < ring_buffer_size_; node_idx++) {
                (node_idx < ring_buffer_size_ - 1) ? ring_buffer_[node_idx].next = (&ring_buffer_[node_idx + 1])
                                                   : ring_buffer_[node_idx].next = &(ring_buffer_[0]);
            }
            // Initialize head and tail ptrs to start of ring buffer.
            head_ = ring_buffer_;
            tail_ = ring_buffer_;
        }
        void push(std::function<void()>&& task) {
            // Stall condition: this push will update the tail (wptr)
            // to match the location of head (rptr). The current push can
            // thus overwrite data that's being read. Stall until head
            // has progressed (data has been read).
            // A stall is only required when the ring_buffer_ backing the queue
            // is full. Realistically, this should never happen, given the size
            while (tail_.load()->next == head_.load());
            tail_.load()->data = std::move(task);
            tail_.store(tail_.load()->next);
        }
        std::function<void()>&& pop() {
            CThreadPool::WorkerQueue::Node* old_head = pop_head();
            return std::move(old_head->data);
        }
        bool empty() const { return head_.load() == tail_.load(); }

    private:
        struct Node {
            std::function<void()> data;
            Node* next = nullptr;
        };

        std::atomic<Node*> head_;
        std::atomic<Node*> tail_;

        Node* pop_head() {
            Node* old_head = head_.load();
            if (old_head == tail_.load()) {
                return nullptr;  // Queue is empty
            }
            head_.store(old_head->next);
            return old_head;
        }
        // Statically allocated ring buffer containing
        // node objects, which contain handles to data
        // and another node object to traverse ring buffer.
        const static uint32_t ring_buffer_size_ = 32768;
        Node ring_buffer_[ring_buffer_size_];
    };

    // Worker threads backing the pool
    std::vector<std::thread> workers_;
    // Task queue
    WorkerQueue tasks_;
    // Mutex to synchronize workers when reading
    // from task queue
    std::mutex mutex_;
    // Counting Semaphore used by main thread to
    // notify workers when a task is available
    std::counting_semaphore<> task_semaphore_{0};
    // Atomic counter used by workers to notify
    // main thread when all tasks are complete
    std::atomic<int> counter_ = 0;
    bool shutdown_;
    // Track the application thread owning this pool
    // Currently only support a single producer to
    // thread pool. Additional funcitonality for multi-producer
    // use cases can be added in future.
    std::atomic<std::thread::id> application_thread_id_;
};

}  // namespace detail

std::shared_ptr<ThreadPool> create_boost_thread_pool(int num_threads) {
    return std::make_shared<detail::BoostThreadPool>(num_threads);
}

std::shared_ptr<ThreadPool> create_custom_thread_pool(int num_threads) {
    return std::make_shared<detail::CThreadPool>(num_threads);
}
}  // namespace tt::tt_metal
