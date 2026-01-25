#ifndef __CORE_TYPES_H__
#define __CORE_TYPES_H__

#include <stdint.h>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

struct range64
{
	u64 baseOffset : 32;
	u64 count : 32;
};

constexpr i32 INVALID_IDX = -1;

template<typename T>
inline bool IsIndexValid( T idx ) { return T( INVALID_IDX ) != idx; }

template<typename T>
concept Number32BitsMax = ( sizeof( T ) <= 4 );

#endif // !__CORE_TYPES_H__
