#ifndef __HP_MATERIAL_H__
#define __HP_MATERIAL_H__

#include "core_types.h"
#include "hp_math.h"

enum alpha_mode : u8
{
	ALPHA_MODE_OPAQUE,
	ALPHA_MODE_MASK,
	ALPHA_MODE_BLEND,
};

struct material_info
{
	float4 baseColFactor;
	float metallicFactor;
	float roughnessFactor;
	float alphaCutoff;
	//float pad1;
	float3 emissiveFactor;
	//float pad2;

	u16 baseColorIdx;
	u16 metallicRoughnessIdx;
	u16 normalIdx;
	u16 occlusionIdx;
	u16 emissiveIdx;
	u16 samplerIdx;

	alpha_mode alphaMode;
};

#endif // !__HP_MATERIAL_H__
