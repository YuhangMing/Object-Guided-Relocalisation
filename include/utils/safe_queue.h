#ifndef XUTILS_DATA_STRUCT_SAFE_QUEUE_H
#define XUTILS_DATA_STRUCT_SAFE_QUEUE_H

#include <queue>
#include <mutex>

namespace xutils
{

template <class T>
class SafeQueue
{
public:
    void push(const T &);
    bool pop(T &val);
    void clear();

private:
    std::mutex resource;
    std::queue<T> queue;
};

template <class T>
void SafeQueue<T>::push(const T &val)
{
    std::lock_guard<std::mutex> lock(resource);
    queue.push(val);
}

template <class T>
bool SafeQueue<T>::pop(T &val)
{
    std::lock_guard<std::mutex> lock(resource);

    if (queue.size() == 0)
        return false;

    val = queue.front();
    queue.pop();
    return true;
}

template <class T>
void SafeQueue<T>::clear()
{
    std::lock_guard<std::mutex> lock(resource);
    while (queue.size() != 0)
        queue.pop();
}

} // namespace xutils

#endif