#include <meshoptimizer.h>

#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include <span>
#include <ranges>

#include <ankerl/unordered_dense.h>

#include "core_types.h"
#include "hp_error.h"

// NOTE/TODO: float types are fwd def in mesh to be shared ! must use an internal folder or smth for math
#include "hp_math.h"
#include "hp_mesh.h"

#include "hp_material.h"
#include "hp_bvh.h"
#include "hp_encoding.h"
#include "hp_bcn_compression.h"
#include "hp_serialization.h"
#include "mikkt_space.h"

#include "range_utils.h"

#include "gltf_loader.h"

#include "hp_types_internal.h"

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

__meshopt_meshlets MakeMeshletsRaytracing( 
	std::span<const float3> pos, 
	std::span<const u32> indices, 
	cluster_raytracing clusterConfig 
) {
	const u64 indexCount = std::size( indices );

	// NOTE ( meshoptimizer ): use minTriangles to compute worst case bound
	const u64 maxMeshletCount = meshopt_buildMeshletsBound( indexCount, clusterConfig.maxVertices, clusterConfig.minTriangles );
	std::vector<meshopt_Meshlet> meshlets( maxMeshletCount );
	std::vector<u32> mletVtx( indexCount );
	std::vector<u8> mletTris( indexCount );

	const u64 meshletCount = meshopt_buildMeshletsSpatial( 
		std::data( meshlets ), std::data( mletVtx ), std::data( mletTris ), std::data( indices ), std::size( indices ), 
		&pos[ 0 ].x, std::size( pos ), sizeof( pos[ 0 ] ), clusterConfig.maxVertices, 
		clusterConfig.minTriangles, clusterConfig.maxTriangles, clusterConfig.fillWeight);

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

auto PackMeshletsRaytracing( const raw_mesh& rawMesh )
{
	__meshopt_meshlets meshlets = MakeMeshletsRaytracing( rawMesh.pos, rawMesh.indices, cluster_raytracing{} );

	std::span<const float3> pos = rawMesh.pos;
	std::span<const float3> norm = rawMesh.normals;
	std::span<const float4> tan = rawMesh.tans;
	std::span<const float2> uvs = rawMesh.uvs;

	std::vector<rt_meshlet> packedMeshlets;
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

		std::vector<packed_vtx> packedAttrs( std::size( mletNormStream ) );
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

		rt_meshlet rtMeshlet = {
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

auto PackVertexAttributes( std::span<const float3> normals, std::span<const float4> tans, std::span<const float2> uvs )
{
	u64 vtxCount = std::size( normals );

	HP_ASSERT( ( vtxCount == std::size( tans ) ) && ( vtxCount == std::size( uvs ) ) );

	std::vector<packed_vtx> packedAttrs;
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
	std::span<const u32> indices = rawMesh.indices;
	auto it = std::ranges::max_element( indices );
	HP_ASSERT( it != std::cend( indices ) );
	HP_ASSERT( *it <= u16( -1 ) );
	HP_ASSERT( rawMesh.materialIdx <= i32( u16( -1 ) ) );
}

packed_mesh PackMesh( const raw_mesh& rawMesh, bvh_builder& bvhBuilder )
{
	std::vector<packed_vtx> packedAttrs;
	
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
	std::vector<packed_vtx> meshVtxAttr( std::size( packedAttrs ) );
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
	std::vector<gpu_bvh2_node> globalTlasBuffer;
	std::vector<gpu_bvh2_node> globalClasBuffer;
	std::vector<rt_meshlet_info> meshletInfoBuffer;
	std::vector<position_t> globalVertexPosBuffer;
	std::vector<packed_vtx> globalPackedVertexBuffer;
	std::vector<u8> globalTriangleBuffer;

	range64 AppendMeshlets( const std::ranges::forward_range auto& meshlets )
	{
		HP_ASSERT( std::size( globalVertexPosBuffer ) == std::size( globalPackedVertexBuffer ) );
		HP_ASSERT( std::size( globalVertexPosBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalTriangleBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalClasBuffer ) < u32( -1 ) );

		u64 totalVertexCount = 0;
		u64 totalTrianlgeCount = 0;
		for( const rt_meshlet& m : meshlets )
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

		for( const rt_meshlet& m : meshlets )
		{
			rt_meshlet_info info = {
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

struct world_data
{
	std::vector<mesh_instance> instances;
	std::vector<gpu_bvh2_node> globalTlasBuffer;
	std::vector<gpu_bvh2_node> globalBlasBuffer;
	std::vector<position_t> globalVertexPosBuffer;
	std::vector<packed_vtx> globalPackedVertexBuffer;
	std::vector<index_t> globalTriangleBuffer;

	mesh_desc AppendMesh( const packed_mesh& mesh )
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

	world_data worldData = {};

	bvh_builder bvhBuilder = {};

	std::vector<packed_mesh> packedMeshes;
	packedMeshes.reserve( std::size( rawMeshes ) );

	ankerl::unordered_dense::map<u32, mesh_desc> meshDescMap;

	for( u64 mi : std::views::iota( 0u, std::size( rawMeshes ) ) )
	{
		const raw_mesh& m = rawMeshes[ mi ];

		ValidateRawMeshAssumptions( m );

		packed_mesh packed = PackMesh( m, bvhBuilder );
		packedMeshes.emplace_back( packed );

		mesh_desc meshDesc = worldData.AppendMesh( packed );
		meshDescMap.emplace( mi, meshDesc );
	}

	std::vector<mesh_instance> instances;
	instances.reserve( std::size( rawNodes ) );
	for( const raw_node& n : rawNodes )
	{
		if( INVALID_IDX == n.meshIdx ) continue;

		auto[ aabbMin, aabbMax ] = TransformAABB( 
			packedMeshes[ n.meshIdx ].aabb, n.toWorld.t, n.toWorld.r, n.toWorld.s );
		auto it = meshDescMap.find( n.meshIdx );
		HP_ASSERT( it != std::cend( meshDescMap ) );

		instances.push_back( {
			.toWorld = n.toWorld,
			.aabbMin = aabbMin,
			.aabbMax = aabbMax,
			.meshDesc = it->second,
			.materialIdx = packedMeshes[ n.meshIdx ].materialIdx
		} );
	}

	auto tlasAabbsView = InstanceTlasAabbView( instances );
	bvh_output tlasOutput = bvhBuilder.BuildBvhOverPrimitives( tlasAabbsView );
	worldData.globalTlasBuffer = std::move( tlasOutput.gpuNodes );

	HP_ASSERT( std::size( instances ) == std::size( tlasOutput.primitiveIndices ) );
	worldData.instances.reserve( std::size( instances ) );
	auto permutedInstances = PermutedView( instances, tlasOutput.primitiveIndices );
	std::ranges::copy( permutedInstances, std::back_inserter( worldData.instances ) );

	std::vector<u8> blob = worldData.HellPackSerialize();
	WriteFileBinary( "D:/3d models/nightclub_futuristic_pub_ambience_asset.hllp", blob );

	if constexpr( CHECK_SERIALIZATION_RESULT )
	{
		auto read = ReadFileBinary( "D:/3d models/nightclub_futuristic_pub_ambience_asset.hllp" );
		hellpack_view hellpackView = { read };

		byte_view bufs[] = {
			MakeByteView( worldData.instances ),
			MakeByteView( worldData.globalTlasBuffer ),
			MakeByteView( worldData.globalBlasBuffer ),
			MakeByteView( worldData.globalVertexPosBuffer ),
			MakeByteView( worldData.globalPackedVertexBuffer ),
			MakeByteView( worldData.globalTriangleBuffer ),
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

