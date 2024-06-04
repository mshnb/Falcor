/**
    Compute shader for sampling in lights.
*/

#include "Utils/Math/MathConstants.slangh"

__exported import Scene.Shading;

import Rendering.Lights.EnvMapSampler;
import Utils.Sampling.UniformSampleGenerator;

static const bool kUseEnvLight = USE_ENV_LIGHT;

RWTexture2D<float4> gPosW;
RWTexture2D<float4> gNormW;
RWTexture2D<float4> gEmissive;

uniform EnvMapSampler envS;

struct LightSample
{
    float3 position;
    float3 normal;
    float3 emissive;
//    float pdf;
//    uint materialID;
};

bool samplePointLight(const LightData light, out LightSample ls)
{
    // Get the position and normal.
    ls.position = light.posW;
    ls.normal = float3(-100,-100,-100); // set it to a special value
    ls.emissive = light.intensity;
    return true;
}

bool sampleDirectionalLight(const LightData light, out LightSample ls)
{
    ls.position = float3(-1,-1,-1);
    ls.normal = light.dirW;
    ls.emissive = light.intensity;
    return true;
}

bool sampleRectAreaLight(const LightData light,const float2 u,  out LightSample ls)
{
    float3 pos = float3(u.x * 2.f - 1.f, u.y * 2.f - 1.f, 0.f);

    // Apply model to world transformation matrix.
    ls.position = mul(light.transMat, float4(pos, 1.f)).xyz;

    // Setup world space normal.
    // TODO: normalW is not correctly oriented for mesh instances that have flipped triangle winding.
    ls.normal = normalize(mul(light.transMatIT, float4(0.f, 0.f, 1.f, 0.f)).xyz);
    ls.emissive = light.intensity;
    return true;
}

bool sampleSphereAreaLight(const LightData light, const float2 u,out LightSample ls)
{
    float phi = M_2PI * u.y;
    float cosTheta = 1.0f - 2.0f * u.x;
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float3 pos = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    // Apply model to world transformation matrix.
    ls.position = mul(light.transMat, float4(pos, 1.f)).xyz;

    // Setup world space normal.
    ls.normal = normalize(mul(light.transMatIT, float4(pos, 0.f)).xyz);
    ls.emissive = light.intensity;
    return true;
}

bool sampleDiscAreaLight(const LightData light, const float2 u,out LightSample ls)
{
    float2 p;
    float r = sqrt(u.x);
    float phi = M_2PI * u.y;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    float3 pos = float3(p, 0.f);

    // Transform to world space.
    ls.position = mul(light.transMat, float4(pos, 1.f)).xyz;

    // Setup world space normal.
    ls.normal = normalize(mul(light.transMatIT, float4(0.f, 0.f, 1.f, 0.f)).xyz);
    ls.emissive = light.intensity;
    return true;
}

bool sampleDistantLight(const LightData light, const float2 u,out LightSample ls)
{
    // TODO
    ls.position = float3(-1.f, -1.f, -1.f);

    // Sample direction.
    float cosTheta = light.cosSubtendedAngle;
    float z = u.x * (1.f - cosTheta) + cosTheta;
    float r = sqrt(1.f - z*z);
    float phi = M_2PI * u.y;
    float3 dir =  float3(r * cos(phi), r * sin(phi), z);

    ls.normal = -dir;
    ls.emissive = light.intensity;
    return true;
}

bool sampleLight<S : ISampleGenerator>(const LightData light, inout S sg, out LightSample ls)
{
    // Sample the light based on its type: point, directional, or area.
    switch (light.type)
    {
    case LightType::Point:
        return samplePointLight(light, ls);
    case LightType::Directional:
        return sampleDirectionalLight(light, ls);
    case LightType::Rect:
        return sampleRectAreaLight(light, sampleNext2D(sg), ls);
    case LightType::Sphere:
        return sampleSphereAreaLight(light, sampleNext2D(sg), ls);
    case LightType::Disc:
        return sampleDiscAreaLight(light, sampleNext2D(sg), ls);
    case LightType::Distant:
        return sampleDistantLight(light, sampleNext2D(sg), ls);
    default:
        ls = {};
        return false; // Should not happen
    }
}

void generateAnalyticLightSample(inout UniformSampleGenerator sg,const LightData light, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    // Sample analytic light source selected uniformly from the light list.
    // TODO: Sample based on estimated contributions as pdf.

    // Sample local light source.
    sampleLight(light,sg,ls);
}

float3 sample_triangle(float2 u)
{
    float su = sqrt(u.x);
    float2 b = float2(1.f - su, u.y * su);
    return float3(1.f - b.x - b.y, b.x, b.y);
}

bool sampleTriangle(const uint triangleIndex, const float2 u, out LightSample ls)
{
    ls = {};
    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
    // Sample the triangle uniformly.
    const float3 barycentrics = sample_triangle(u);
    ls.position = tri.getPosition(barycentrics);
    ls.normal = tri.normal;
    float2 uv = tri.getTexCoord(barycentrics);
    ls.emissive = gScene.materials.evalEmissive(tri.materialID, uv);
    //ls.materialID = tri.materialID;
    //ls.pdf = 1/tri.area;
    return true;
}

[numthreads(256, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    uint2 pixel = threadID.xy;
    UniformSampleGenerator sg=UniformSampleGenerator(pixel,1000);
    uint lightCount = gScene.getLightCount();
    uint index = 0;
    for(uint i = 0; i < lightCount; i++){
        LightData lightData= gScene.getLight(i);
        float2 randomXY = sampleNext2D(sg);
        LightSample ls;
        generateAnalyticLightSample(sg,lightData,ls);
        gPosW[uint2(threadID.x ,index)] = float4(ls.position,1.0);
        gNormW[uint2(threadID.x ,index)] = float4(ls.normal,1.0);
        gEmissive[uint2(threadID.x  ,index)] = float4(ls.emissive,1.0);
        index++;
    }

    uint emissiveCount = gScene.lightCollection.meshCount;
    if(emissiveCount > 0){
        for(uint i = 0; i<emissiveCount;i++){
            uint meshIndex = i;
            uint triangleCount = gScene.lightCollection.meshData[meshIndex].triangleCount;
            uint triangleOffset = gScene.lightCollection.meshData[meshIndex].triangleOffset;
            uint triangleIndex = min(uint(sampleNext1D(sg) * triangleCount), triangleCount - 1) + triangleOffset;
            LightSample ls;
            sampleTriangle(triangleIndex,sampleNext2D(sg),ls);
            //ls.pdf = ls.pdf / float(triangleCount);
            gPosW[uint2(threadID.x ,index)] = float4(ls.position,1.0);
            gNormW[uint2(threadID.x ,index)] = float4(ls.normal,1.0);
            gEmissive[uint2(threadID.x  ,index)] = float4(ls.emissive,1.0);
            index++;
        }
    }

    if(kUseEnvLight){
        EnvMapSample lightSample;

        envS.sample(sampleNext2D(sg), lightSample);
        gPosW[uint2(threadID.x ,index)] = float4(lightSample.dir,1.0);
        gNormW[uint2(threadID.x ,index)] = float4(-lightSample.dir,1.0);
        gEmissive[uint2(threadID.x  ,index)] = float4(lightSample.Le,1.0);
    }
}
