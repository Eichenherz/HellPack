#ifndef __RANGE_UTILS_H__
#define __RANGE_UTILS_H__

#include <ranges>
#include <algorithm>
#include <vector>

template<typename T, typename Idx>
inline auto PermutedView( std::vector<T>& src, const std::vector<Idx>& remap )
{
	return remap | std::views::transform( [&]( Idx oldIdx ) -> T& { return src[ oldIdx ]; } );
}

template<typename T, typename Idx>
inline auto PermutedView( const std::vector<T>& src, const std::vector<Idx>& remap )
{
	return remap | std::views::transform( [&]( Idx oldIdx ) -> const T& { return src[ oldIdx ]; } );
}

inline auto PermutedView( 
	const std::ranges::random_access_range auto& src,
	const std::ranges::random_access_range auto& remap 
) {
	return remap | std::views::transform( [&] ( auto oldIdx ) { return src[ ( u32 ) oldIdx ]; } );
}

template<typename TriIdx, typename PrimIdx>
inline auto PermuteTrianglesByPrimitiveRemap( const std::vector<TriIdx>& oldIdx, const std::vector<PrimIdx>& primitiveIndices ) 
{
	u64 triangleCount = std::size( primitiveIndices );
	HP_ASSERT( ( triangleCount * 3 ) == std::size( oldIdx ) );

	std::vector<TriIdx> newIdx( std::size( oldIdx ) );
	for( u64 ti = 0; ti < triangleCount; ++ti )
	{
		u64 oldTi = primitiveIndices[ ti ];
		u64 src = 3ull * oldTi;
		u64 dst = 3ull * ti;

		newIdx[ dst + 0 ] = oldIdx[ src + 0 ];
		newIdx[ dst + 1 ] = oldIdx[ src + 1 ];
		newIdx[ dst + 2 ] = oldIdx[ src + 2 ];
	}

	return newIdx;
}

template<typename Idx>
inline auto BuildVertexRemapFromPermutedIndices( const std::vector<Idx>& permutedIndices, u64 vtxCount )
{
	constexpr auto INVALID_IDX = u32( -1 );

	HP_ASSERT( Idx( -1 ) >= vtxCount );

	std::vector<Idx> remap( vtxCount, INVALID_IDX );
	u32 next = 0;

	for( Idx idx : permutedIndices )
	{
		Idx oldV = idx;
		if( INVALID_IDX == remap[ oldV ] )
		{
			remap[ oldV ] = next++;
		}
	}

	return remap;
}


inline bool ByteEqual( std::span<const u8> a, std::span<const u8> b )
{
	bool sizeEq = std::size( a ) == std::size( b );
	return sizeEq && ( std::memcmp( std::data( a ), std::data( b ), std::size( a ) ) == 0 );
}

#endif // !__RANGE_UTILS_H__
