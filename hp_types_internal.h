#ifndef __HP_TYPES_INTERNAL_H__
#define __HP_TYPES_INTERNAL_H__

#include "core_types.h"

#include <vector>
#include <string>

#include "hp_math.h"

struct raw_mesh
{
	std::string         name;
	std::vector<float3> pos;
	std::vector<float3> normals;
	std::vector<float4> tans;
	std::vector<float2> uvs;
	std::vector<u32>    indices;
	i32                 materialIdx;
};

enum class image_channels_t : u8
{
	UNKNOWN = 0,
	R = 1,
	RG = 2,
	RGB = 3,
	RGBA = 4
};

enum class image_bit_depth_t : u8
{
	UNKNOWN = 0,
	B8  = 8,
	B16 = 16,
	B32 = 32
};

enum class image_pixel_type : u8
{
	UNKNOWN = 0,
	UBYTE,
	USHORT,
	FLOAT32
};

struct image_metadata
{
	u16 width;
	u16 height;
	image_channels_t component;
	image_bit_depth_t  bits;
	image_pixel_type pixelType;
};

struct raw_image_view
{
	std::span<const u8> data;
	image_metadata metadata;
};

struct packed_mesh
{
	std::vector<float3> positions;
	std::vector<vertex_attrs> attrs;
	std::vector<u16> indices;
	std::vector<gpu_bvh2_node> blas;
	aabb_t<float3> aabb;
	u16 materialIdx;
};



#endif // !__HP_TYPES_INTERNAL_H__
