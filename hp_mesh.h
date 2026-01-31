#ifndef __HP_MESH_H__
#define __HP_MESH_H__

#include "core_types.h"
#include "hp_math.h"

#include <bit>
#include <string>
#include <vector>

struct alignas( 16 ) packed_trs
{
	float3 t;
	float pad0;
	float4 r;
	float3 s;
	float pad1;
};

struct raw_node
{
	packed_trs toWorld;
	i32 meshIdx;
};

constexpr bool LEFT_HANDED = true;
static_assert( LEFT_HANDED );

struct raw_mesh
{
	std::string name;
	std::vector<float3> pos;
	std::vector<float3> normals;
	std::vector<float4> tans;
	std::vector<float2> uvs;
	std::vector<u32> indices;
	i32 materialIdx;
};

struct rt_meshlet_info
{
	float3	aabbMin;
	float3	aabbMax;

	u32    vertexOffset;
	u32    triangleOffset;
	u8     vertexCount;
	u8     triangleCount;
};

struct packed_vtx
{
	float2 octNormal;
	float tanAngle;
	float u, v;
	u8 tanSign;
};

struct rt_meshlet
{
	std::vector<float3> positions;
	std::vector<packed_vtx> packedAttrs;
	std::vector<u8> triangles;
	float3	aabbMin;
	float3	aabbMax;
};

struct cluster_regular
{
	float coneWeight = 0.8f;
	u16 maxVertices = 64;
	u16 maxTriangles = 128;
};

struct cluster_raytracing
{
	float fillWeight = 0.5f;
	u16 maxVertices = 64;
	u16 minTriangles = 16;
	u16 maxTriangles = 64;
};

// NOTE: alias bc this can refer both to a node or a leaf
using bvh2_node_ref32 = u32;

constexpr u32 BVH_INVALID_REF = 0xffffffffu;
constexpr u32 BVH2_LEAF_BIT   = 0x80000000u;
constexpr u32 BVH2_NODE_MASK  = 0x7fffffffu;

// NOTE: example-leaf: base = bits0..28 (index into prim_ids[]) countMinus1 = bits29..30 (0..3 => count 1..4)
constexpr u32 MIN_LEAF_PRIM_COUNT = 1;
constexpr u32 MAX_LEAF_PRIM_COUNT = 4;
static_assert( MAX_LEAF_PRIM_COUNT >= 1 && MAX_LEAF_PRIM_COUNT <= 8 );

constexpr u32 BVH2_LEAF_COUNT_BITS = std::bit_width( MAX_LEAF_PRIM_COUNT );
constexpr u32 BVH2_LEAF_COUNT_SHIFT = 31u - BVH2_LEAF_COUNT_BITS;
constexpr u32 BVH2_LEAF_BASE_MASK = ( 1u << BVH2_LEAF_COUNT_SHIFT ) - 1u;
constexpr u32 BVH2_LEAF_COUNT_MASK = ( ( 1u << BVH2_LEAF_COUNT_BITS ) - 1u ) << BVH2_LEAF_COUNT_SHIFT;

inline bool  Bvh2IsLeaf( bvh2_node_ref32 ref ) { return ( ref & BVH2_LEAF_BIT ) != 0u; }
inline u32	 Bvh2NodeIdx( bvh2_node_ref32 ref ) { return ( ref & BVH2_NODE_MASK ); }
inline u32	 Bvh2LeafBase( bvh2_node_ref32 ref ) { return ( ref & BVH2_LEAF_BASE_MASK ); }
inline u32	 Bvh2LeafCount( bvh2_node_ref32 ref ) 
{ 
	return ( ( ref & BVH2_LEAF_COUNT_MASK ) >> BVH2_LEAF_COUNT_SHIFT ) + 1u; 
}
inline bool  Bvh2RefIsInvalid( bvh2_node_ref32 ref )
{
	return BVH_INVALID_REF == ref;
}

struct alignas( 64 ) gpu_bvh2_node
{
	std::array<float3, 2> min;
	std::array<float3, 2> max;
	std::array<bvh2_node_ref32, 2> childIdx; 
};

struct instance
{
	packed_trs toWorld;
	bvh2_node_ref32 clasBvhRoot;
	u32 clasNodeCount;
	u32 baseMeshletOffset;
	u32 meshletCount;
	i32 materialIdx;
};

#endif // !__HP_MESH_H__
