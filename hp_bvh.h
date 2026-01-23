#ifndef __HP_BVH_H__
#define __HP_BVH_H__

#include "core_types.h"
#include "hp_math.h"
#include "hp_types.h"

#include <bit>

struct alignas( 64 ) gpu_bvh2_node
{
	std::array<float3,2> min;
	std::array<float3,2> max;
	std::array<u32,2> childIdx; 
};

constexpr u32 BVH_INVALID_REF = 0xffffffffu;
constexpr u32 BVH2_LEAF_BIT   = 0x80000000u;
constexpr u32 BVH2_NODE_MASK  = 0x7fffffffu;

// NOTE: example-leaf: base = bits0..28 (index into prim_ids[]) countMinus1 = bits29..30 (0..3 => count 1..4)
constexpr u32 MIN_LEAF_PRIM_COUNT = 1;
constexpr u32 MAX_LEAF_PRIM_COUNT = 4; // 1..8, stored as (count-1)

static_assert( MAX_LEAF_PRIM_COUNT >= 1 && MAX_LEAF_PRIM_COUNT <= 8 );

constexpr u32 BVH2_LEAF_COUNT_BITS = std::bit_width( MAX_LEAF_PRIM_COUNT );
constexpr u32 BVH2_LEAF_COUNT_SHIFT = 31u - BVH2_LEAF_COUNT_BITS;
constexpr u32 BVH2_LEAF_BASE_MASK = ( 1u << BVH2_LEAF_COUNT_SHIFT ) - 1u;
constexpr u32 BVH2_LEAF_COUNT_MASK = ( ( 1u << BVH2_LEAF_COUNT_BITS ) - 1u ) << BVH2_LEAF_COUNT_SHIFT;

inline bool  GpuBvh2IsLeaf( u32 ref ) { return ( ref & BVH2_LEAF_BIT ) != 0u; }
inline u32	 GpuBvh2NodeIdx( u32 ref ) { return ( ref & BVH2_NODE_MASK ); }

inline u32	 GpuBvh2LeafBase( u32 ref ) { return ( ref & BVH2_LEAF_BASE_MASK ); }
inline u32	 GpuBvh2LeafCount( u32 ref ) { return ( ( ref & BVH2_LEAF_COUNT_MASK ) >> BVH2_LEAF_COUNT_SHIFT ) + 1u; }

inline u32   MakeGpuBvh2NodeRef( u32 nodeIndex ) { return nodeIndex & BVH2_NODE_MASK; }
inline u32   MakeGpuBvh2LeafRef( u32 basePrimIds, u32 count )
{
	assert( count >= 1 && count <= MAX_LEAF_PRIM_COUNT );

	u32 c = ( count - 1u ) & ( ( 1u << BVH2_LEAF_COUNT_BITS ) - 1u );

	return BVH2_LEAF_BIT | ( basePrimIds & BVH2_LEAF_BASE_MASK ) | ( c << BVH2_LEAF_COUNT_SHIFT );
}


#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/node.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/sweep_sah_builder.h>

// NOTE: not thread safe, use one instance per thread !
struct bvh_builder
{
	using scalar  = float;
	using vec3    = bvh::v2::Vec<scalar, 3>;
	using bbox_t  = bvh::v2::BBox<scalar, 3>;
	using node_t  = bvh::v2::Node<scalar, 3>;

	std::vector<bbox_t> aabbs;
	std::vector<vec3> centers;

	inline static vec3 FromFloat3( float3 in )
	{
		return { in.x, in.y, in.z };
	}
	inline static float3 ToFloat3( vec3 in )
	{
		return { in[ 0 ], in[ 1 ], in[ 2 ] };
	}
	inline static aabb_t<float3> GetAabb( bbox_t in )
	{
		return { .min = ToFloat3( in.min ), .max = ToFloat3( in.max ) };
	}

	inline bvh::v2::Bvh<node_t> BuildSweptSahFromRtMeshlets( std::span<const rt_meshlet> meshlets )
	{
		using namespace bvh::v2;

		const u64 meshletCount = std::size( meshlets );

		aabbs.resize( 0 );
		centers.resize( 0 );
		aabbs.reserve( meshletCount );
		centers.reserve( meshletCount );

		for( const auto& m : meshlets )
		{
			bbox_t aabb = { FromFloat3( m.aabbMin ), FromFloat3( m.aabbMax ) };
			aabbs.push_back( aabb );
			centers.push_back( aabb.get_center() );
		}

		SweepSahBuilder<node_t>::Config config = { 
			.min_leaf_size = MIN_LEAF_PRIM_COUNT, .max_leaf_size = MAX_LEAF_PRIM_COUNT };
		return SweepSahBuilder<node_t>::build( aabbs, centers, config );
	}

	static u32 EmitChildDfs( const bvh::v2::Bvh<node_t>& bvh, u32 madChildId, std::vector<gpu_bvh2_node>& gpuNodes )
	{
		const auto& c = bvh.nodes[ madChildId ];

		if( c.index.is_leaf() )
		{
			return MakeGpuBvh2LeafRef( c.index.first_id(), c.index.prim_count() );
		}

		u32 first = c.index.first_id();
		u32 c0Mad = first + 0u;
		u32 c1Mad = first + 1u;

		const auto& c0 = bvh.nodes[ c0Mad ];
		const auto& c1 = bvh.nodes[ c1Mad ];

		aabb_t<float3> c0Aabb = GetAabb( c0.get_bbox() );
		aabb_t<float3> c1Aabb = GetAabb( c1.get_bbox() );

		// NOTE: preorder allocation => subtree-contiguous DFS layout
		u32 outId = std::size( gpuNodes );
		gpuNodes.emplace_back();

		gpu_bvh2_node& o = gpuNodes[ outId ];
		o.min = { c0Aabb.min, c1Aabb.min };
		o.max = { c0Aabb.max, c1Aabb.max };

		// NOTE: recurse left then right => DFS layout
		o.childIdx[ 0 ] = EmitChildDfs( bvh, c0Mad, gpuNodes );
		o.childIdx[ 1 ] = EmitChildDfs( bvh, c1Mad, gpuNodes );

		return MakeGpuBvh2NodeRef( outId );
	}

	inline auto LinearizeBvhDfsPacked_Recursive( const bvh::v2::Bvh<node_t>& bvh )
	{
		std::vector<gpu_bvh2_node> gpuNodes;
		gpuNodes.reserve( std::size( bvh.nodes ) );

		const auto& root = bvh.nodes[ 0 ];

		// NOTE: Degenerate BVH root is leaf
		if( root.index.is_leaf() )
		{
			aabb_t<float3> aabb = GetAabb( root.get_bbox() );
			u32 leafRef = MakeGpuBvh2LeafRef( root.index.first_id(), root.index.prim_count() );
			gpuNodes.push_back( {
				.min = { aabb.min },
				.max = { aabb.max },
				.childIdx = { leafRef, BVH_INVALID_REF }
								} );

			return gpuNodes;
		}

		u32 first = root.index.first_id();
		u32 c0Mad = first + 0u;
		u32 c1Mad = first + 1u;

		const auto& c0 = bvh.nodes[ c0Mad ];
		const auto& c1 = bvh.nodes[ c1Mad ];

		aabb_t<float3> c0Aabb = GetAabb( c0.get_bbox() );
		aabb_t<float3> c1Aabb = GetAabb( c1.get_bbox() );

		gpuNodes.emplace_back();

		gpu_bvh2_node& rootGpu = gpuNodes[ 0 ];
		rootGpu.min = { c0Aabb.min, c1Aabb.min };
		rootGpu.max = { c0Aabb.max, c1Aabb.max };

		rootGpu.childIdx[ 0 ] = EmitChildDfs( bvh, c0Mad, gpuNodes );
		rootGpu.childIdx[ 1 ] = EmitChildDfs( bvh, c1Mad, gpuNodes );

		return gpuNodes;
	}
};

#endif // !__HP_BVH_H__
