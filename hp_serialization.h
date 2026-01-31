#ifndef __HP_SERIALIZATION_H__
#define __HP_SERIALIZATION_H__

#include "core_types.h"
#include "hp_error.h"

#include <span>
#include <algorithm>
#include <ranges>

constexpr u32 HELLPACK_VERSION = 1;
constexpr u64 HELLPACK_MAGIC =
	( u64( 'H' ) )       |
	( u64( 'E' ) << 8 )  |
	( u64( 'L' ) << 16 ) |
	( u64( 'L' ) << 24 ) |
	( u64( 'P' ) << 32 ) |
	( u64( 'A' ) << 40 ) |
	( u64( 'C' ) << 48 ) |
	( u64( 'K' ) << 56 );

constexpr u64 GPU_BUFFER_ALIGNEMNT = 64;

inline u64 AlignUp( u64 x, u64 a ) { return ( x + ( a - 1 ) ) & ~( a - 1 ); }

struct byte_range_t
{
	u64 baseOffset;
	u64 count;
};

template<typename T>
struct typed_view
{
	const T* ptr = nullptr;
	u32 count = 0;

	constexpr const T* data()  const { return ptr; }
	constexpr u32      size()  const { return count; }
	constexpr const T* begin() const { return ptr; }
	constexpr const T* end()   const { return ptr + count; }
	constexpr std::span<const T> span() const { return { ptr, ( u64 ) count }; }
	constexpr const T& operator[](u32 i) const noexcept
	{
		assert( i < count );
		return ptr[ i ];
	}
};

using byte_view = typed_view<u8>;

template<typename T>
static inline byte_view AsBytes( typed_view<T> v )
{
	static_assert( std::is_trivially_copyable_v<T> );
	return { ( const u8* ) std::data( v ), ( u32 ) std::size( v ) * sizeof( T ) };
}

template<typename T>
inline typed_view<T> MakeTypedView( std::span<const T> s )
{
	return { std::data( s ), ( u32 ) std::size( s ) };
}

template<std::ranges::contiguous_range R>
inline byte_view MakeByteView( const R& r )
{
	using T = std::ranges::range_value_t<R>;

	static_assert( std::is_trivially_copyable_v<T> );
	return { ( const u8* ) std::data( r ), ( u32 ) std::size( r ) * sizeof( T ) };
}

enum hellpack_entry_slot : u8
{
	INST = 0,
	TLAS,
	CLAS,
	MLET,
	VPOS,
	PVTX,
	TRI,
	COUNT
};

#pragma pack(push, 1)
struct hellpack_blob_header
{
	u64 magic;
	u32 version;
	u32 pad0;
	u64 fileSizeBytes;

	byte_range_t ranges[ hellpack_entry_slot::COUNT ];
};
#pragma pack(pop)

static_assert( sizeof( hellpack_blob_header ) % 8 == 0 );

inline auto MakeHellpackBlob( std::span<const byte_view> bufs )
{
	hellpack_blob_header h = {
		.magic = HELLPACK_MAGIC,
		.version = HELLPACK_VERSION
	};

	u64 cursor = AlignUp( sizeof( hellpack_blob_header ), GPU_BUFFER_ALIGNEMNT );

	for( u64 i : std::views::iota( 0u, std::size( bufs ) ) ) 
	{
		cursor = AlignUp( cursor, GPU_BUFFER_ALIGNEMNT );

		u64 thisBuffSize = std::size( bufs[ i ] );
		h.ranges[ i ] = { .baseOffset = cursor, .count = thisBuffSize };

		cursor += thisBuffSize;
	}
	cursor = AlignUp( cursor, GPU_BUFFER_ALIGNEMNT );

	h.fileSizeBytes = cursor;

	// NOTE: pad to zero automatically
	std::vector<u8> blob( h.fileSizeBytes, 0 );
	std::memcpy( std::data( blob ), &h, sizeof( h ) );

	auto MemcpyLambda = [&] ( u64 i )
	{
		const u64 offset = h.ranges[ i ].baseOffset;
		const u64 size = h.ranges[ i ].count;

		HP_ASSERT( offset + size <= std::size( blob ) );

		if( size )
		{
			std::memcpy( std::data( blob ) + offset, std::data( bufs[ i ] ), size );
		}
	};
	std::ranges::for_each( std::views::iota( 0u, std::size( bufs ) ), MemcpyLambda );

	return blob;
}

struct hellpack_view
{
	const u8* base;
	u64 sizeInBytes;
	const hellpack_blob_header* h;

	hellpack_view( std::span<const u8> blob )
	{
		HP_ASSERT( std::size( blob ) >= sizeof( hellpack_blob_header ) );

		base = std::data( blob );
		sizeInBytes = std::size( blob );
		h = ( const hellpack_blob_header* ) ( base );

		HP_ASSERT( h->magic == HELLPACK_MAGIC );
		HP_ASSERT( h->version == HELLPACK_VERSION );
		HP_ASSERT( h->fileSizeBytes == sizeInBytes );
	}

	byte_view Bytes( hellpack_entry_slot s ) const
	{
		const u64 offset = h->ranges[ s ].baseOffset;
		const u64 size = h->ranges[ s ].count;

		HP_ASSERT( offset + size <= sizeInBytes );

		return { base + offset, ( u32 ) size };
	}

	template<typename T>
	typed_view<T> Typed( hellpack_entry_slot s ) const
	{
		const u64 offset = h->ranges[ s ].baseOffset;
		const u64 size = h->ranges[ s ].count;

		HP_ASSERT( offset % alignof( T ) == 0 );
		HP_ASSERT( offset + size <= sizeInBytes );
		HP_ASSERT( ( size % sizeof( T ) ) == 0 );

		return { ( const T* ) ( base + offset ), ( u32 ) ( size / sizeof( T ) ) };
	}
};

#endif // !__HP_SERIALIZATION_H__
