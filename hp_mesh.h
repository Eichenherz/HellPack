#ifndef __HP_MESH_H__
#define __HP_MESH_H__

#include "core_types.h"
#include "hp_bvh.h"

#include <vector>

struct float2;
struct float3;
struct float4;

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


struct meshlet_info
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
	float3 pos;
	float2 octNormal;
	float tanAngle;
	float u, v;
	u8 tanSign;
};

struct vertex_attrs
{
	float2 octNormal;
	float tanAngle;
	float u, v;
	u8 tanSign;
};

struct meshlet
{
	float3	aabbMin;
	float3	aabbMax;

	u32 vtxOffset;
	u32 triOffset;

	u32 vtxCount;
	u32 triCount;
};


struct world_node
{
	packed_trs toWorld;
	u16 meshIdx;
	u16 materialIdx;
};

struct rt_cluster
{
	std::vector<float3> positions;
	std::vector<vertex_attrs> packedAttrs;
	std::vector<u8> triangles;
	float3	aabbMin;
	float3	aabbMax;
};

struct rt_clustered_instance
{
	packed_trs toWorld;
	float3 aabbMin;
	float3 aabbMax;
	bvh2_node_ref32 clasBvhRoot;
	u32 clasNodeCount;
	u32 baseMeshletOffset;
	u32 meshletCount;
	u16 materialIdx;
};

#endif // !__HP_MESH_H__
