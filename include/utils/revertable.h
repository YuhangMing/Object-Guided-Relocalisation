#ifndef XUTILS_DATA_STRUCT_REVERTABLE_H
#define XUTILS_DATA_STRUCT_REVERTABLE_H

namespace fusion
{

template <class T>
class Revertable
{
public:
    Revertable() = default;
    Revertable(const T &);
    Revertable &operator=(const T &);
    void revert();
    T get() const;

private:
    T curr;
    T prev;
};

template <class T>
Revertable<T>::Revertable(const T &val)
{
    curr = val;
}

template <class T>
Revertable<T> &Revertable<T>::operator=(const T &val)
{
    prev = curr;
    curr = val;
    return *this;
}

template <class T>
void Revertable<T>::revert()
{
    curr = prev;
}

template <class T>
T Revertable<T>::get() const
{
    return curr;
}

} // namespace fusion

#endif