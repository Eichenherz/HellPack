#include <meshoptimizer.h>

#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include <span>
#include <ranges>
#include <type_traits>

#include <ankerl/unordered_dense.h>

#include "core_types.h"
#include "hp_error.h"

// NOTE/TODO: float types are fwd def in mesh to be shared ! must use an internal folder or smth for math
#include "hp_math.h"
#include "hp_mesh.h"

#include "hp_material.h"
#include "hp_bvh_builder.h"
#include "hp_encoding.h"
#include "hp_bcn_compression.h"
#include "hp_serialization.h"
#include "mikkt_space.h"

#include "range_utils.h"

#include "gltf_loader.h"

#include "hp_types_internal.h"

struct meshlet_config
{
	float coneWeight = 0.8f;
	u16 maxVertices = 64;
	u16 maxTriangles = 128;
};

struct rt_cluster_config
{
	float fillWeight = 0.5f;
	u16 maxVertices = 64;
	u16 minTriangles = 16;
	u16 maxTriangles = 64;
};

template<typename T>
concept IsMeshletCfg = std::same_as<T, meshlet_config>;

template<typename T>
concept IsRtClusterCfg = std::same_as<T, rt_cluster_config>;

template<typename T>
concept IsClusterConfig = IsMeshletCfg<T> || IsRtClusterCfg<T>;

// TODO: no inplace remap !
void ReindexAndOptimizeMesh( raw_mesh& rawMesh )
{
	meshopt_Stream attrStreams[] = {
		{ .data = std::data( rawMesh.pos ), .size = std::size( rawMesh.pos ), .stride = sizeof( rawMesh.pos[ 0 ] ) },
		{ .data = std::data( rawMesh.normals ), .size = std::size( rawMesh.normals ), .stride = sizeof( rawMesh.normals[ 0 ] ) },
		{ .data = std::data( rawMesh.tans ), .size = std::size( rawMesh.tans ), .stride = sizeof( rawMesh.tans[ 0 ] ) },
		{ .data = std::data( rawMesh.uvs ), .size = std::size( rawMesh.uvs ), .stride = sizeof( rawMesh.uvs[ 0 ] ) },
	};
	std::vector<u32>& indices = rawMesh.indices;

	const u64 vtxCount = std::size( rawMesh.pos );
	const u64 idxCount = std::size( indices );

	std::vector<u32> remap( vtxCount );
	u64 newVtxCount = meshopt_generateVertexRemapMulti(
		std::data( remap ), std::data( indices ), idxCount, vtxCount, attrStreams, std::size( attrStreams ) );

	HP_ASSERT( newVtxCount <= vtxCount );
	if( newVtxCount != vtxCount )
	{
		meshopt_remapIndexBuffer( std::data( indices ), std::data( indices ), idxCount, std::data( remap ) );
		meshopt_remapVertexBuffer( std::data( rawMesh.pos ), std::data( rawMesh.pos ), vtxCount, 
								   sizeof( rawMesh.pos[ 0 ] ), std::data( remap ) );
		rawMesh.pos.resize( newVtxCount );
		meshopt_remapVertexBuffer( std::data( rawMesh.normals ), std::data( rawMesh.normals ), vtxCount, 
								   sizeof( rawMesh.normals[ 0 ] ), std::data( remap ) );
		rawMesh.normals.resize( newVtxCount );
		meshopt_remapVertexBuffer( std::data( rawMesh.tans ), std::data( rawMesh.tans ), vtxCount, 
								   sizeof( rawMesh.tans[ 0 ] ), std::data( remap ) );
		rawMesh.tans.resize( newVtxCount );
		meshopt_remapVertexBuffer( std::data( rawMesh.uvs ), std::data( rawMesh.uvs ), vtxCount, 
								   sizeof( rawMesh.uvs[ 0 ] ), std::data( remap ) );
		rawMesh.uvs.resize( newVtxCount );
	}

	meshopt_optimizeVertexCache( std::data( indices ), std::data( indices ), idxCount, newVtxCount );

	meshopt_optimizeVertexFetch( std::data( rawMesh.pos ), std::data( indices ), idxCount, std::data( rawMesh.pos ), 
								 newVtxCount, sizeof( rawMesh.pos[ 0 ] ) );
	meshopt_optimizeVertexFetch( std::data( rawMesh.normals ), std::data( indices ), idxCount, std::data( rawMesh.normals ), 
								 newVtxCount, sizeof( rawMesh.normals[ 0 ] ) );
	meshopt_optimizeVertexFetch( std::data( rawMesh.tans ), std::data( indices ), idxCount, std::data( rawMesh.tans ), 
								 newVtxCount, sizeof( rawMesh.tans[ 0 ] ) );
	meshopt_optimizeVertexFetch( std::data( rawMesh.uvs ), std::data( indices ), idxCount, std::data( rawMesh.uvs ), 
								 newVtxCount, sizeof( rawMesh.uvs[ 0 ] ) );
}

struct __meshopt_meshlets
{
	std::vector<meshopt_Meshlet> info;
	std::vector<u32> vertices;
	std::vector<u8> triangles;
};

template<IsClusterConfig Cfg>
__meshopt_meshlets MeshoptMakeClusters( 
	std::span<const float3> pos, 
	std::span<const u32> indices, 
	Cfg cfg 
) { 
	const u64 indexCount = std::size( indices );

	u64 triangleBound = cfg.maxTriangles;
	if constexpr( IsRtClusterCfg<Cfg> )
	{
		// NOTE( meshoptimizer ): use minTriangles to compute worst case bound
		triangleBound = cfg.minTriangles;
	}
	
	const u64 maxMeshletCount = meshopt_buildMeshletsBound( indexCount, cfg.maxVertices, triangleBound );
	std::vector<meshopt_Meshlet> meshlets( maxMeshletCount );
	std::vector<u32> mletVtx( indexCount );
	std::vector<u8> mletTris( indexCount );

	u64 meshletCount;
	if constexpr( IsMeshletCfg<Cfg> )
	{
		meshletCount = meshopt_buildMeshlets(
			&meshlets[ 0 ], &mletVtx[ 0 ], &mletTris[ 0 ], &indices[ 0 ], std::size( indices ),
			&pos[ 0 ].x, std::size( pos ), sizeof( pos[ 0 ] ), cfg.maxVertices, cfg.maxTriangles, cfg.coneWeight );
	}
	else if constexpr( IsRtClusterCfg<Cfg> )
	{
		meshletCount = meshopt_buildMeshletsSpatial(
			&meshlets[ 0 ], &mletVtx[ 0 ], &mletTris[ 0 ], &indices[ 0 ], std::size( indices ), &pos[ 0 ].x, 
			std::size( pos ), sizeof( pos[ 0 ] ), cfg.maxVertices, cfg.minTriangles, cfg.maxTriangles, cfg.fillWeight );
	}

	const meshopt_Meshlet& last = meshlets[ meshletCount - 1 ];

	meshlets.resize( meshletCount );
	mletVtx.resize( ( u64 ) last.vertex_offset + last.vertex_count );
	mletTris.resize( ( u64 ) last.triangle_offset + ( ( ( u64 ) last.triangle_count * 3 + 3 ) & ~3 ) );

	for( const meshopt_Meshlet& m : meshlets )
	{
		meshopt_optimizeMeshlet( &mletVtx[ m.vertex_offset ], &mletTris[ m.triangle_offset ],
								 m.triangle_count, m.vertex_count );
	}

	return { 
		.info = std::move( meshlets ), 
		.vertices = std::move( mletVtx ), 
		.triangles = std::move( mletTris ) 
	};
}

template<typename T> 
inline auto GetMeshletLocalAttrStream( 
	std::span<const T> meshAttrStream, 
	std::span<const u32> mletVtx, 
	u64 mletVtxOffset, 
	u64 mletVtxCount 
){
	std::vector<T> localStream( mletVtxCount );
	for( u64 vi = 0; vi < std::size( localStream ); ++vi )
	{
		localStream[ vi ] = meshAttrStream[ mletVtx[ vi + mletVtxOffset ] ];
	}

	return localStream;
}


std::vector<meshlet> PackMeshlets( const raw_mesh& rawMesh )
{
	__meshopt_meshlets meshlets = MeshoptMakeClusters( rawMesh.pos, rawMesh.indices, meshlet_config{} );

	std::span<const float3> pos = rawMesh.pos;
	std::span<const float3> norm = rawMesh.normals;
	std::span<const float4> tan = rawMesh.tans;
	std::span<const float2> uvs = rawMesh.uvs;

	std::vector<meshlet> packedMeshlets;
	packedMeshlets.reserve( std::size( meshlets.info ) );

	for( const meshopt_Meshlet& m : meshlets.info )
	{
		auto firstTriangleIt = std::cbegin( meshlets.triangles ) + m.triangle_offset;
		std::vector<u8> triangles = { firstTriangleIt, firstTriangleIt + m.triangle_count };

		std::vector<float3> mletPosStream = GetMeshletLocalAttrStream( pos, meshlets.vertices, m.vertex_offset, m.vertex_count );

		auto mletNormStream = GetMeshletLocalAttrStream( norm, meshlets.vertices, m.vertex_offset, m.vertex_count );
		auto mletTanStream = GetMeshletLocalAttrStream( tan, meshlets.vertices, m.vertex_offset, m.vertex_count );
		auto mletUvStream = GetMeshletLocalAttrStream( uvs, meshlets.vertices, m.vertex_offset, m.vertex_count );

		std::vector<packed_vtx> packedVtx( std::size( mletPosStream ) );
		for( u64 vi = 0; vi < m.vertex_count; ++vi )
		{
			float3 p = mletPosStream[ vi ];
			float3 n = mletNormStream[ vi ];
			float4 t = mletTanStream[ vi ];
			float2 uv = mletUvStream[ vi ];
			float2 octNormal = OctaNormalEncode( n );
			float tanAngle = EncodeTanToAngle( n, { t.x,t.y,t.z } );
			u8 tanSign = ( -1.0f == t.w ) ? 1 : 0;

			packedVtx[ vi ] = {
				.pos = p,
				.octNormal = octNormal, 
				.tanAngle = tanAngle, 
				.u = uv.x, .v = uv.y, 
				.tanSign = tanSign 
			};
		}

		const aabb_t<float3> aabb = ComputeAabb( mletPosStream );

		meshlet rtMeshlet = {
			.vertices = std::move( packedVtx ), 
			.triangles = std::move( triangles ),
			.aabbMin = aabb.min, 
			.aabbMax = aabb.max 
		};
		packedMeshlets.push_back( std::move( rtMeshlet ) );
	}

	return packedMeshlets;
}

std::vector<rt_cluster> PackClustersRaytracing( const raw_mesh& rawMesh )
{
	__meshopt_meshlets meshlets = MeshoptMakeClusters( rawMesh.pos, rawMesh.indices, rt_cluster_config{} );

	std::span<const float3> pos = rawMesh.pos;
	std::span<const float3> norm = rawMesh.normals;
	std::span<const float4> tan = rawMesh.tans;
	std::span<const float2> uvs = rawMesh.uvs;

	std::vector<rt_cluster> packedMeshlets;
	packedMeshlets.reserve( std::size( meshlets.info ) );

	for( const meshopt_Meshlet& m : meshlets.info )
	{
		auto firstTriangleIt = std::cbegin( meshlets.triangles ) + m.triangle_offset;
		std::vector<u8> triangles = { firstTriangleIt, firstTriangleIt + m.triangle_count };

		std::vector<float3> mletPosStream = GetMeshletLocalAttrStream( pos, meshlets.vertices, m.vertex_offset, m.vertex_count );

		const aabb_t<float3> aabb = ComputeAabb( mletPosStream );

		auto mletNormStream = GetMeshletLocalAttrStream( norm, meshlets.vertices, m.vertex_offset, m.vertex_count );
		auto mletTanStream = GetMeshletLocalAttrStream( tan, meshlets.vertices, m.vertex_offset, m.vertex_count );
		auto mletUvStream = GetMeshletLocalAttrStream( uvs, meshlets.vertices, m.vertex_offset, m.vertex_count );

		std::vector<vertex_attrs> packedAttrs( std::size( mletNormStream ) );
		for( u64 vi = 0; vi < m.vertex_count; ++vi )
		{
			float3 n = mletNormStream[ vi ];
			float4 t = mletTanStream[ vi ];
			float2 uv = mletUvStream[ vi ];
			float2 octNormal = OctaNormalEncode( n );
			float tanAngle = EncodeTanToAngle( n, { t.x,t.y,t.z } );
			u8 tanSign = ( -1.0f == t.w ) ? 1 : 0;

			packedAttrs[ vi ] = {
				.octNormal = octNormal, 
				.tanAngle = tanAngle, 
				.u = uv.x, .v = uv.y, 
				.tanSign = tanSign 
			};
		}

		rt_cluster rtMeshlet = {
			.positions = std::move( mletPosStream ), 
			.packedAttrs = std::move( packedAttrs ), 
			.triangles = std::move( triangles ),
			.aabbMin = aabb.min, 
			.aabbMax = aabb.max 
		};
		packedMeshlets.push_back( std::move( rtMeshlet ) );
	}

	return packedMeshlets;
}

std::vector<vertex_attrs> PackVertexAttributes( 
	std::span<const float3> normals, 
	std::span<const float4> tans, 
	std::span<const float2> uvs 
) {
	u64 vtxCount = std::size( normals );

	HP_ASSERT( ( vtxCount == std::size( tans ) ) && ( vtxCount == std::size( uvs ) ) );

	std::vector<vertex_attrs> packedAttrs;
	packedAttrs.resize( vtxCount );

	for( u64 vi = 0; vi < vtxCount; ++vi )
	{
		float3 n = normals[ vi ];
		float4 t = tans[ vi ];
		float2 uv = uvs[ vi ];
		float2 octNormal = OctaNormalEncode( n );
		float tanAngle = EncodeTanToAngle( n, { t.x,t.y,t.z } );
		u8 tanSign = ( -1.0f == t.w ) ? 1 : 0;

		packedAttrs[ vi ] = {
			.octNormal = octNormal, 
			.tanAngle = tanAngle, 
			.u = uv.x, .v = uv.y, 
			.tanSign = tanSign 
		};
	}

	return packedAttrs;
}

void ValidateRawMeshAssumptions( const raw_mesh& rawMesh )
{
	auto it = std::ranges::max_element( rawMesh.indices );
	HP_ASSERT( it != std::cend( rawMesh.indices ) );
	HP_ASSERT( *it <= u16( -1 ) );
	HP_ASSERT( rawMesh.materialIdx <= i32( u16( -1 ) ) );
}

packed_mesh PackMesh( const raw_mesh& rawMesh, bvh_builder& bvhBuilder )
{
	std::vector<vertex_attrs> packedAttrs;
	
	if( !std::size( rawMesh.tans ) )
	{
		std::vector<float4> tans = ComputeMikkTSpaceTangentsInplace( rawMesh );
		packedAttrs = PackVertexAttributes( rawMesh.normals, tans, rawMesh.uvs );
	}
	else
	{
		packedAttrs = PackVertexAttributes( rawMesh.normals, rawMesh.tans, rawMesh.uvs );
	}

	const auto& positions = rawMesh.pos;
	const auto& indices = rawMesh.indices;

	std::vector<float3> meshPos( std::size( positions ) );
	std::vector<vertex_attrs> meshVtxAttr( std::size( packedAttrs ) );
	std::vector<u16> meshIdx( std::size( indices ) );
	
	std::vector<gpu_bvh2_node> blas;
	aabb_t<float3> aabb = {};

	const bool buildBvhThresholdReached = std::size( indices ) > 3 * MAX_LEAF_PRIM_COUNT;
	if( buildBvhThresholdReached )
	{
		auto triAabbView = TriangleAabbView( positions, indices );
		bvh_output bvh = bvhBuilder.BuildBvhOverPrimitives( triAabbView );

		blas = bvh.gpuNodes;
		aabb = bvh.topLevelAabb;

		std::vector<u32> permutedIndices = PermuteTrianglesByPrimitiveRemap( indices, bvh.primitiveIndices );
		std::vector<u32> remap = BuildVertexRemapFromPermutedIndices( permutedIndices, std::size( positions ) );

		meshopt_remapIndexBuffer( 
			std::data( permutedIndices ), std::data( permutedIndices ), std::size( permutedIndices ), std::data( remap ) );

		u64 vtxCount = std::size( meshPos );
		meshopt_remapVertexBuffer( 
			std::data( meshPos ), std::data( positions ), vtxCount, sizeof( positions[ 0 ] ), std::data( remap ) );
		meshopt_remapVertexBuffer( 
			std::data( meshVtxAttr ), std::data( packedAttrs ), vtxCount, sizeof( packedAttrs[ 0 ] ), std::data( remap ) );

		auto u16IndicesView = permutedIndices | std::views::transform( [] ( u32 idx ) { return ( u16 ) idx; } );
		std::ranges::copy( u16IndicesView, std::begin( meshIdx ) );
	}
	else
	{
		auto u16IndicesView = rawMesh.indices | std::views::transform( [] ( u32 idx ) { return ( u16 ) idx; } );
		
		std::ranges::copy( positions, std::begin( meshPos ) );
		std::ranges::copy( packedAttrs, std::begin( meshVtxAttr ) );
		std::ranges::copy( u16IndicesView, std::begin( meshIdx ) );

		aabb = ComputeAabb( positions );
	}
	
	return {
		.positions = std::move( meshPos ),
		.attrs = std::move( meshVtxAttr ),
		.blas = std::move( blas ),
		.aabb = aabb,
		.materialIdx = ( u16 ) rawMesh.materialIdx 
	};
}

using position_t = float3;
using index_t = u16;

struct clustered_world_data
{
	std::vector<clustered_instance> instances;
	std::vector<meshlet_info> meshletInfoBuffer;
	std::vector<packed_vtx> globalMeshletVertexBuffer;
	std::vector<u8> globalMeshletTriangleBuffer;

	range64 AppendMeshlets( const std::ranges::forward_range auto& meshlets )
	{
		HP_ASSERT( std::size( globalMeshletVertexBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalMeshletTriangleBuffer ) < u32( -1 ) );

		u64 totalVertexCount = 0;
		u64 totalTrianlgeCount = 0;
		for( const meshlet& m : meshlets )
		{
			const u64 vertexCount = std::size( m.vertices );
			const u64 triangleCount = std::size( m.triangles );

			HP_ASSERT( ( vertexCount <= u8( -1 ) ) && ( triangleCount <= u8( -1 ) ) );

			totalVertexCount += vertexCount;
			totalTrianlgeCount += triangleCount;
		}
		globalMeshletVertexBuffer.reserve( std::size( globalMeshletVertexBuffer ) + totalVertexCount );
		globalMeshletTriangleBuffer.reserve( std::size( globalMeshletTriangleBuffer ) + totalTrianlgeCount );

		HP_ASSERT( std::size( meshletInfoBuffer ) <= u32( -1 ) );
		const u32 baseMeshletOffset = ( u32 ) std::size( meshletInfoBuffer );
		meshletInfoBuffer.reserve( baseMeshletOffset + std::size( meshlets ) );

		for( const meshlet& m : meshlets )
		{
			meshlet_info info = {
				.aabbMin = m.aabbMin,
				.aabbMax = m.aabbMax,
				.vertexOffset = ( u32 ) std::size( globalMeshletVertexBuffer ),
				.triangleOffset = ( u32 ) std::size( globalMeshletTriangleBuffer ),
				.vertexCount = ( u8 ) std::size( m.vertices ),
				.triangleCount = ( u8 ) std::size( m.triangles )
			};

			std::ranges::copy( m.vertices, std::back_inserter( globalMeshletVertexBuffer ) );
			std::ranges::copy( m.triangles, std::back_inserter( globalMeshletTriangleBuffer ) );
			meshletInfoBuffer.push_back( info );
		}

		HP_ASSERT( std::size( meshlets ) <= u32( -1 ) );
		return { .baseOffset = baseMeshletOffset, .count = ( u32 ) std::size( meshlets ) };
	}

	inline std::vector<u8> HellPackSerialize()
	{
		byte_view bufs[] = {
			MakeByteView( instances ),
			MakeByteView( meshletInfoBuffer ),
			MakeByteView( globalMeshletVertexBuffer ),
			MakeByteView( globalMeshletTriangleBuffer ),
		};

		HP_ASSERT( std::size( bufs ) == hellpack_entry_slot::COUNT );

		return MakeHellpackBlob( bufs );
	}
};

struct rt_clustered_world_data
{
	std::vector<rt_clustered_instance> instances;
	std::vector<gpu_bvh2_node> globalTlasBuffer;
	std::vector<gpu_bvh2_node> globalClasBuffer;
	std::vector<meshlet_info> meshletInfoBuffer;
	std::vector<position_t> globalVertexPosBuffer;
	std::vector<vertex_attrs> globalPackedVertexBuffer;
	std::vector<u8> globalTriangleBuffer;

	range64 AppendMeshlets( const std::ranges::forward_range auto& meshlets )
	{
		HP_ASSERT( std::size( globalVertexPosBuffer ) == std::size( globalPackedVertexBuffer ) );
		HP_ASSERT( std::size( globalVertexPosBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalTriangleBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalClasBuffer ) < u32( -1 ) );

		u64 totalVertexCount = 0;
		u64 totalTrianlgeCount = 0;
		for( const rt_cluster& m : meshlets )
		{
			const u64 vertexCount = std::size( m.positions );
			const u64 triangleCount = std::size( m.triangles );

			HP_ASSERT( vertexCount == std::size( m.packedAttrs ) );
			HP_ASSERT( ( vertexCount <= u8( -1 ) ) && ( triangleCount <= u8( -1 ) ) );

			totalVertexCount += vertexCount;
			totalTrianlgeCount += triangleCount;
		}
		globalVertexPosBuffer.reserve( std::size( globalVertexPosBuffer ) + totalVertexCount );
		globalPackedVertexBuffer.reserve( std::size( globalPackedVertexBuffer ) + totalVertexCount );
		globalTriangleBuffer.reserve( std::size( globalTriangleBuffer ) + totalTrianlgeCount );

		HP_ASSERT( std::size( meshletInfoBuffer ) <= u32( -1 ) );
		const u32 baseMeshletOffset = ( u32 ) std::size( meshletInfoBuffer );
		meshletInfoBuffer.reserve( baseMeshletOffset + std::size( meshlets ) );

		for( const rt_cluster& m : meshlets )
		{
			meshlet_info info = {
				.aabbMin = m.aabbMin,
				.aabbMax = m.aabbMax,
				.vertexOffset = ( u32 ) std::size( globalVertexPosBuffer ),
				.triangleOffset = ( u32 ) std::size( globalTriangleBuffer ),
				.vertexCount = ( u8 ) std::size( m.positions ),
				.triangleCount = ( u8 ) std::size( m.triangles )
			};

			std::ranges::copy( m.positions, std::back_inserter( globalVertexPosBuffer ) );
			std::ranges::copy( m.packedAttrs, std::back_inserter( globalPackedVertexBuffer ) );
			std::ranges::copy( m.triangles, std::back_inserter( globalTriangleBuffer ) );
			meshletInfoBuffer.push_back( info );
		}

		HP_ASSERT( std::size( meshlets ) <= u32( -1 ) );
		return { .baseOffset = baseMeshletOffset, .count = ( u32 ) std::size( meshlets ) };
	}

	inline std::vector<u8> HellPackSerialize()
	{
		byte_view bufs[] = {
			MakeByteView( instances ),
			MakeByteView( globalTlasBuffer ),
			MakeByteView( globalClasBuffer ),
			MakeByteView( meshletInfoBuffer ),
			MakeByteView( globalVertexPosBuffer ),
			MakeByteView( globalPackedVertexBuffer ),
			MakeByteView( globalTriangleBuffer ),
		};

		HP_ASSERT( std::size( bufs ) == hellpack_entry_slot::COUNT );

		return MakeHellpackBlob( bufs );
	}
};

struct rt_world_data
{
	std::vector<mesh_instance> instances;
	std::vector<gpu_bvh2_node> globalTlasBuffer;
	std::vector<gpu_bvh2_node> globalBlasBuffer;
	std::vector<position_t> globalVertexPosBuffer;
	std::vector<vertex_attrs> globalPackedVertexBuffer;
	std::vector<index_t> globalTriangleBuffer;

	rt_mesh_desc AppendMesh( const packed_mesh& mesh )
	{
		HP_ASSERT( std::size( globalVertexPosBuffer ) == std::size( globalPackedVertexBuffer ) );
		HP_ASSERT( std::size( globalVertexPosBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalTriangleBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalBlasBuffer ) < u32( -1 ) );

		HP_ASSERT( std::size( mesh.blas ) < u16( -1 ) );
		HP_ASSERT( std::size( mesh.positions ) < u16( -1 ) );
		HP_ASSERT( std::size( mesh.indices ) < u16( -1 ) );

		u32 baseBvhNodeOffset = std::size( globalBlasBuffer );
		u32 baseVertexOffset = std::size( globalVertexPosBuffer );
		u32 baseIndexOffset = std::size( globalTriangleBuffer );

		u16 blasNodeCount = std::size( mesh.blas );
		u16 vertexCount = std::size( mesh.positions );
		u16 indexCount = std::size( mesh.indices );

		globalBlasBuffer.reserve( baseBvhNodeOffset + blasNodeCount );
		globalVertexPosBuffer.reserve( baseVertexOffset + vertexCount );
		globalPackedVertexBuffer.reserve( baseVertexOffset + vertexCount );
		globalTriangleBuffer.reserve( baseIndexOffset + indexCount );

		std::ranges::copy( mesh.blas, std::back_inserter( globalBlasBuffer ) );
		std::ranges::copy( mesh.positions, std::back_inserter( globalVertexPosBuffer ) );
		std::ranges::copy( mesh.attrs, std::back_inserter( globalPackedVertexBuffer ) );
		std::ranges::copy( mesh.indices, std::back_inserter( globalTriangleBuffer ) );

		return { 
			.bvhRoot = baseBvhNodeOffset,
			.baseVertexOffset = baseVertexOffset,
			.baseIndexOffset = baseIndexOffset,
			.bvhNodeCount = blasNodeCount,
			.vertexCount = vertexCount,
			.indexCount = indexCount
		};
	}

	inline std::vector<u8> HellPackSerialize()
	{
		byte_view bufs[] = {
			MakeByteView( instances ),
			MakeByteView( globalTlasBuffer ),
			MakeByteView( globalBlasBuffer ),
			MakeByteView( globalVertexPosBuffer ),
			MakeByteView( globalPackedVertexBuffer ),
			MakeByteView( globalTriangleBuffer ),
		};

		HP_ASSERT( std::size( bufs ) == hellpack_entry_slot::COUNT );

		return MakeHellpackBlob( bufs );
	}
};

inline auto MeshletAabbView( const std::ranges::forward_range auto& meshlets )
{
	return meshlets | std::views::transform( 
		[] ( const auto& m ) { return aabb_t<float3>{ .min = m.aabbMin, .max = m.aabbMax }; } );
};

inline auto InstanceTlasAabbView( const std::ranges::forward_range auto& instances )
{
	return instances | std::views::transform( 
		[] ( const auto& i ) 
		{ 
			return TransformAABB( i.aabbMin, i.aabbMax, i.toWorld.t, i.toWorld.r, i.toWorld.s );
		} );
};

struct texture_compression_batch
{
	ankerl::unordered_dense::map<u32,u32> imgViewToTexMap;
	std::vector<bcn_compression_result> bcn;

	std::vector<bcn_compression_job> jobs;
	u32 size;

	inline void ProcessItem( u32 jobIdx )
	{
		const bcn_compression_job& job = jobs[ jobIdx ];
		bcn_compression_result& result = bcn[ jobIdx ];
		// NOTE: these allocate memory !
		result = CompressRGBA8ToBCn( job );
	}
};

texture_compression_batch 
PrepareBcnCompressionBatch( std::span<const material_info> materials, std::span<const raw_image_view> imageViews )
{
	std::vector<bcn_compression_job> jobs;
	ankerl::unordered_dense::map<u32,u32> imgViewToTexMap;

	auto ProcessImageView = [&] ( u16 idx, bc_format_t fmt )
	{
		if( !IsIndexValid( idx ) ) return;
		if( imgViewToTexMap.find( idx ) != std::cend( imgViewToTexMap ) ) return;

		const raw_image_view& imgView = imageViews[ idx ];
		jobs.push_back( bcn_compression_job{
			.rgba8 = imgView.data, 
			.width = imgView.metadata.width, 
			.height = imgView.metadata.height, 
			.format = fmt
		} );

		imgViewToTexMap.emplace( idx, std::size( jobs ) - 1 );
	};

	for( const material_info& material : materials )
	{
		ProcessImageView( material.baseColorIdx, bc_format_t::BC7_RGBA );
		ProcessImageView( material.normalIdx, bc_format_t::BC5_RG );
		ProcessImageView( material.metallicRoughnessIdx, bc_format_t::BC7_RGBA );
		//ProcessImageView( material.occlusionIdx, bc_format_t::BC7_RGBA );
		// NOTE: currently not suporting ambient occlusion which must be packed into MR
		HP_ASSERT( !IsIndexValid( material.occlusionIdx ) );
		ProcessImageView( material.emissiveIdx, bc_format_t::BC7_RGBA );
	}

	u32 batchSize = std::size( jobs );
	std::vector<bcn_compression_result> bcn( batchSize );
	return {
		.imgViewToTexMap = std::move( imgViewToTexMap ),
		.bcn = std::move( bcn ),
		.jobs = std::move( jobs ),
		.size = batchSize
	};
}

// NOTE: RunBatch and Join must always be called in pairs !!!!
struct batch_executor
{
	std::vector<std::jthread> threads;
	std::atomic<u32> jobDoneCounter;

	// NOTE: fetch add will increase past the batchSize
	void RunBatch( auto& BatchProcessor, u32 workerCount, u32 batchSize ) 
	{
		auto WorkerLoop = [this, BatchProcessor, batchSize] ()
		{
			for( ;; )
			{
				u32 jobIdx = jobDoneCounter.fetch_add( 1, std::memory_order_relaxed );
				if( jobIdx >= batchSize ) break;

				BatchProcessor( jobIdx );
			}
		};

		jobDoneCounter = { 0 };
		threads.clear();
		for( u32 ti = 0; ti < workerCount; ++ti )
		{
			threads.emplace_back( WorkerLoop );
		}
	}

	void Join()
	{
		for( auto& t : threads ) t.join();
	}
};

inline void WriteFileBinary( const char* path, std::span<const u8> bytes )
{
	FILE* f = nullptr;
	HP_ASSERT( ::fopen_s( &f, path, "wb" ) == 0 );

	u64 written = ::fwrite( std::data( bytes ), 1, std::size( bytes ), f );
	HP_ASSERT( std::size( bytes ) == written );

	i32 rc = ::fclose( f );
	HP_ASSERT( rc == 0 );
}

inline auto ReadFileBinary( const char* path )
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

constexpr bool CHECK_SERIALIZATION_RESULT = true;

struct mesh_desc
{
	float3 aabbMin;
	float3 aabbMax;
	u32 baseMeshletOffset;
	u16 meshletCount;
	u16 materialIdx;
};

int main()
{
	const std::string gltfFilePath = "D:/3d models/nightclub_futuristic_pub_ambience_asset.glb";
	HP_ASSERT( fs::exists( gltfFilePath ) );

	gltf_loader gltf = { gltfFilePath };

	// TODO: ensure we keep the same indexing as tinygltf provides !!!!
	std::vector<raw_node> rawNodes = gltf.ProcessNodes();
	std::vector<raw_mesh> rawMeshes = gltf.ProcessMeshes();
	std::vector<sampler_config> sampler = gltf.ProcessSamplers();
	std::vector<material_info> materials = gltf.ProcessMaterials();
	std::vector<raw_image_view> imageViews = gltf.ProcessImages();

	//texture_compression_batch bcnBatch = PrepareBcnCompressionBatch( materials, imageViews );
	//
	//batch_executor batchExec = {};
	//auto Proc = [&] ( u32 idx ) { bcnBatch.ProcessItem( idx ); };
	//batchExec.RunBatch( Proc, std::max<u32>( 1, bcnBatch.size / 4 ), bcnBatch.size );

	clustered_world_data worldData = {};

	//bvh_builder bvhBuilder = {};

	std::vector<mesh_desc> meshDesc( std::size( rawMeshes ) );
	for( u64 mi : std::views::iota( 0u, std::size( rawMeshes ) ) )
	{
		raw_mesh& m = rawMeshes[ mi ];

		if( !std::size( m.tans ) )
		{
			m.tans = ComputeMikkTSpaceTangentsInplace( m );
		}

		std::vector<meshlet> meshlets = PackMeshlets( m );
		auto aabbView = meshlets | std::views::transform( 
			[] ( const meshlet& m ) { return aabb_t<float3>{.min = m.aabbMin, .max = m.aabbMax }; } );

		aabb_t<float3> aabb = MergeAabbs( aabbView );

		range64 meshletsRange = worldData.AppendMeshlets( meshlets );

		meshDesc[ mi ] = {
			.aabbMin = aabb.min,
			.aabbMax = aabb.max,
			.baseMeshletOffset = ( u32 ) meshletsRange.baseOffset,
			.meshletCount = ( u16 ) meshletsRange.count,
			.materialIdx = ( u16 ) m.materialIdx
		};
	}

	std::vector<clustered_instance> instances;
	instances.reserve( std::size( rawNodes ) );
	for( const raw_node& n : rawNodes )
	{
		if( INVALID_IDX == n.meshIdx ) continue;

		const mesh_desc& md = meshDesc[ n.meshIdx ];

		instances.push_back( {
			.toWorld = n.toWorld,
			.aabbMin = md.aabbMin,
			.aabbMax = md.aabbMax,
			.baseMeshletOffset = md.baseMeshletOffset,
			.meshletCount = md.meshletCount,
			.materialIdx = md.meshletCount
		} );
	}

	worldData.instances = std::move( instances );

	std::vector<u8> blob = worldData.HellPackSerialize();
	WriteFileBinary( "D:/3d models/nightclub_futuristic_pub_ambience_asset.hllp", blob );

	if constexpr( CHECK_SERIALIZATION_RESULT )
	{
		auto read = ReadFileBinary( "D:/3d models/nightclub_futuristic_pub_ambience_asset.hllp" );
		hellpack_view hellpackView = { read };

		byte_view bufs[] = {
			MakeByteView( worldData.instances ),
			MakeByteView( worldData.meshletInfoBuffer ),
			MakeByteView( worldData.globalMeshletVertexBuffer ),
			MakeByteView( worldData.globalMeshletTriangleBuffer ),
		};

		for( u64 i : std::views::iota( 0u, hellpack_entry_slot::COUNT ) )
		{
			byte_view bv = hellpackView.Bytes( hellpack_entry_slot( i ) );
			HP_ASSERT( ByteEqual( bv, bufs[ i ] ) );
		}
	}
	
	//batchExec.Join();

	return 0;
}

