#ifndef __HP_TYPES_H__
#define __HP_TYPES_H__

#include "core_types.h"
#include "hp_math.h"

#include <vector>

struct alignas( 16 ) packed_trs
{
	float3 t;
	float pad0;
	float4 r;
	float3 s;
	float pad1;
};

constexpr i32 INVALID_IDX = -1;

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

struct rt_packed_meshlet_info
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

#endif // !__HP_TYPES_H__
