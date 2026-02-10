#ifndef __HELL_PACK_H__
#define __HELL_PACK_H__

#include "core_types.h"
#include "hp_error.h"

#include <span>
#include <ranges>

#include "range_utils.h"

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

enum hellpack_entry_type : u8
{
	INST = 0,
	//	TLAS,
	//	BLAS,
	MLET,
	VTX,
	//	VPOS,
	//	PVTX,
	TRI,
	MTRL,
	TEX,
	SAMP,
	COUNT
};

#pragma pack(push, 1)
struct hellpack_entry
{
	u64 offsetInBytes;
	u64 byteCount;
	hellpack_entry_type typeOf;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct hellpack_blob_header
{
	u64 magic;
	u64 fileSizeBytes;
	u64 entryTableOffsetInBytes;
	u32 version;
	u32 entriesCount;
};
#pragma pack(pop)

static_assert( sizeof( hellpack_blob_header ) % 8 == 0 );

// TODO: allow more flexible layout
struct hellpack_view
{
	const u8* base;
	u64 sizeInBytes;
	const hellpack_blob_header* h;
	const hellpack_entry* pEntries;
	u32 entryCount;
	u32 entryTypeToIdxMap[ hellpack_entry_type::COUNT ];


	hellpack_view( std::span<const u8> blob )
	{
		HP_ASSERT( std::size( blob ) >= sizeof( hellpack_blob_header ) );

		base = std::data( blob );
		sizeInBytes = std::size( blob );
		h = ( const hellpack_blob_header* ) ( base );

		HP_ASSERT( h->magic == HELLPACK_MAGIC );
		HP_ASSERT( h->version == HELLPACK_VERSION );
		HP_ASSERT( h->fileSizeBytes == sizeInBytes );
		// NOTE: since we use SoA layout we shouldn't have more than 1 buffer per entry
		HP_ASSERT( h->entriesCount <= std::size( entryTypeToIdxMap ) );

		pEntries = ( const hellpack_entry* ) ( base + h->entryTableOffsetInBytes );
		entryCount = h->entriesCount;

		std::ranges::fill( entryTypeToIdxMap, u32( INVALID_IDX ) );
		for( u64 ei = 0; ei < entryCount; ++ei )
		{
			const hellpack_entry& entry = pEntries[ ei ];
			HP_ASSERT( u32( INVALID_IDX ) == entryTypeToIdxMap[ entry.typeOf ] );
			entryTypeToIdxMap[ entry.typeOf ] = ei;
		}
	}

	byte_view Bytes( hellpack_entry_type s ) const
	{
		u32 entryIdx = entryTypeToIdxMap[ s ];
		if( u32( INVALID_IDX ) == entryIdx ) return {};

		const hellpack_entry& entry = pEntries[ entryIdx ];

		const u64 offset = entry.offsetInBytes;
		const u64 size = entry.byteCount;

		HP_ASSERT( offset + size <= sizeInBytes );

		return { base + offset, ( u32 ) size };
	}

	template<typename T>
	typed_view<T> Typed( hellpack_entry_type s ) const
	{
		u32 entryIdx = entryTypeToIdxMap[ s ];
		if( u32( INVALID_IDX ) == entryIdx ) return {};

		const hellpack_entry& entry = pEntries[ entryIdx ];

		const u64 offset = entry.offsetInBytes;
		const u64 size = entry.byteCount;

		HP_ASSERT( offset % alignof( T ) == 0 );
		HP_ASSERT( offset + size <= sizeInBytes );
		HP_ASSERT( ( size % sizeof( T ) ) == 0 );

		return { ( const T* ) ( base + offset ), ( u32 ) ( size / sizeof( T ) ) };
	}
};

#endif // !__HELL_PACK_H__
