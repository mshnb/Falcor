// Separate GPU pass. Each thread takes one stratified primary-hit surface sample.
cbuffer CB { uint sampleCount; uint seed; uint2 sourceDim; }
Texture2D<float4> gmask;
Texture2D<float4> gdirdepth;
Texture2D<float4> gposition;
Texture2D<float4> gnormal;
Texture2D<float4> galbedo;
Texture2D<float4> gmetalrough;
Texture2D<float4> gemissive;
RWTexture2D<float4> gmaskOut;
RWTexture2D<float4> gdirdepthOut;
RWTexture2D<float4> gpositionOut;
RWTexture2D<float4> gnormalOut;
RWTexture2D<float4> galbedoOut;
RWTexture2D<float4> gmetalroughOut;
RWTexture2D<float4> gemissiveOut;

uint hash32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= sampleCount) return;
    uint cells = (uint)ceil(sqrt((float)sampleCount));
    uint2 cell = uint2(i % cells, i / cells);
    float2 jitter = float2(hash32(i ^ seed) & 65535u, hash32((i + 977u) ^ seed) & 65535u) / 65536.0;
    uint2 p = min(uint2((float2(cell) + jitter) / cells * float2(sourceDim)), sourceDim - 1u);
    uint2 outp = uint2(i, 0);
    gmaskOut[outp] = gmask.Load(int3(p, 0));
    gdirdepthOut[outp] = gdirdepth.Load(int3(p, 0));
    gpositionOut[outp] = gposition.Load(int3(p, 0));
    gnormalOut[outp] = gnormal.Load(int3(p, 0));
    galbedoOut[outp] = galbedo.Load(int3(p, 0));
    gmetalroughOut[outp] = gmetalrough.Load(int3(p, 0));
    gemissiveOut[outp] = gemissive.Load(int3(p, 0));
}
