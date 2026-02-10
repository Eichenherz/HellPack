#ifndef __HP_SERIALIZATION_H__
#define __HP_SERIALIZATION_H__

#include "core_types.h"
#include "hp_error.h"
#include "hp_math.h"
#include "range_utils.h"

#include "hell_pack.h"

#include <span>
#include <algorithm>

constexpr u64 GPU_BUFFER_ALIGNEMNT = 64;

using index_t = u8;

struct hellpack_serializble_buffer
{
	const byte_view data;
	hellpack_entry_type typeOf;

	template<typename T>
	inline hellpack_serializble_buffer( const std::vector<T>& data, hellpack_entry_type typeOf ) 
		: data{ MakeByteView( data ) } , typeOf{ typeOf } {}
};

using hellpack_blob = std::vector<u8>;

inline hellpack_blob MakeHellpackBlob( std::span<const hellpack_serializble_buffer> bufs )
{
	const u32 entriesCount = std::size( bufs );
	// NOTE: since we use SoA layout we shouldn't have more than 1 buffer per entry
	HP_ASSERT( entriesCount <= ( u32 ) hellpack_entry_type::COUNT );

	hellpack_blob_header h = {
		.magic = HELLPACK_MAGIC,
		.version = HELLPACK_VERSION,
		.entriesCount = entriesCount,
	};

	h.entryTableOffsetInBytes = sizeof( h ); // NOTE: for now we place the table right after the header

	std::vector<hellpack_entry> entryTable( entriesCount );
	const u64 entriesTableSizeInBytes = std::size( entryTable ) * sizeof( entryTable[ 0 ] );

	u64 cursor = h.entryTableOffsetInBytes + entriesTableSizeInBytes;

	for( u64 ei = 0; ei < entriesCount; ++ei ) 
	{
		cursor = AlignUp( cursor, GPU_BUFFER_ALIGNEMNT );

		u64 thisBuffSizeInBytes = std::size( bufs[ ei ].data );
		entryTable[ ei ] = { 
			.offsetInBytes = cursor, .byteCount = thisBuffSizeInBytes, .typeOf = bufs[ ei ].typeOf };

		cursor += thisBuffSizeInBytes;
	}
	cursor = AlignUp( cursor, GPU_BUFFER_ALIGNEMNT );

	h.fileSizeBytes = cursor;

	// NOTE: pad to zero automatically
	hellpack_blob blob( h.fileSizeBytes, 0 );
	std::memcpy( std::data( blob ), &h, sizeof( h ) );
	std::memcpy( std::data( blob ) + h.entryTableOffsetInBytes, std::data( entryTable ), entriesTableSizeInBytes );

	for( u64 ei = 0; ei < entriesCount; ++ei )
	{
		const u64 offset = entryTable[ ei ].offsetInBytes;
		const u64 size = entryTable[ ei ].byteCount;

		HP_ASSERT( offset + size <= std::size( blob ) );
		// NOTE: even if we can handle size == 0, there might be an external issue that results in a 0 sized buffer
		HP_ASSERT( 0 != size );
		std::memcpy( std::data( blob ) + offset, std::data( bufs[ ei ].data ), size );
	}

	return blob;
}

inline void WriteFileBinary( const char* path, std::span<const u8> bytes )
{
	FILE* f = nullptr;
	HP_ASSERT( ::fopen_s( &f, path, "wb" ) == 0 );

	u64 written = ::fwrite( std::data( bytes ), 1, std::size( bytes ), f );
	HP_ASSERT( std::size( bytes ) == written );

	i32 rc = ::fclose( f );
	HP_ASSERT( rc == 0 );
}

inline std::vector<u8> ReadFileBinary( const char* path )
{
	FILE* f = nullptr;
	HP_ASSERT( ::fopen_s( &f, path, "rb" ) == 0 );
	HP_ASSERT( f );

	HP_ASSERT( ::fseek( f, 0, SEEK_END ) == 0 );
	i32 sz = ::ftell( f );
	HP_ASSERT( sz >= 0 );
	HP_ASSERT( ::fseek( f, 0, SEEK_SET ) == 0 );

	std::vector<u8> out( sz );
	u64 read = ::fread( std::data( out ), 1, std::size( out ), f );
	HP_ASSERT( std::size( out ) == read );

	HP_ASSERT( ::fclose( f ) == 0 );
	return out;
}

#endif // !__HP_SERIALIZATION_H__
