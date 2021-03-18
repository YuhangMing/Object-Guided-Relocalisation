#ifndef FUSION_MAPPING_HASH_ENTRY_H
#define FUSION_MAPPING_HASH_ENTRY_H

#include "macros.h"
#include "math/matrix_type.h"
#include "math/vector_type.h"

namespace fusion
{

struct FUSION_EXPORT HashEntry
{
    FUSION_HOST_AND_DEVICE inline HashEntry();
    FUSION_HOST_AND_DEVICE inline HashEntry(Vector3i pos, int ptr, int offset);
    FUSION_HOST_AND_DEVICE inline HashEntry(const HashEntry &);
    FUSION_HOST_AND_DEVICE inline HashEntry &operator=(const HashEntry &);
    FUSION_HOST_AND_DEVICE inline bool operator==(const Vector3i &) const;
    FUSION_HOST_AND_DEVICE inline bool operator==(const HashEntry &) const;

    int ptr_;
    int offset_;
    Vector3i pos_;
};

FUSION_HOST_AND_DEVICE inline HashEntry::HashEntry()
    : ptr_(-1), offset_(-1)
{
}

FUSION_HOST_AND_DEVICE inline HashEntry::HashEntry(Vector3i pos, int ptr, int offset)
    : pos_(pos), ptr_(ptr), offset_(offset)
{
}

FUSION_HOST_AND_DEVICE inline HashEntry::HashEntry(const HashEntry &H)
    : pos_(H.pos_), ptr_(H.ptr_), offset_(H.offset_)
{
}

FUSION_HOST_AND_DEVICE inline HashEntry &HashEntry::operator=(const HashEntry &H)
{
    pos_ = H.pos_;
    ptr_ = H.ptr_;
    offset_ = H.offset_;
    return *this;
}

FUSION_HOST_AND_DEVICE inline bool HashEntry::operator==(const Vector3i &pos_) const
{
    return this->pos_ == pos_;
}

FUSION_HOST_AND_DEVICE inline bool HashEntry::operator==(const HashEntry &other) const
{
    return other.pos_ == pos_;
}

} // namespace fusion

#endif