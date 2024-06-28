/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "LightSample.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/AABB.h"

namespace
{
const char kShaderFile[] = "RenderPasses/LightSample/LightSample.cs.slang";

const ChannelList kInputChannels = {
    {"posW", "gPosW", "Position in world space", true /* optional */, ResourceFormat::RGBA32Float},
    {"normW", "gNormW", "Normal in world space", true /* optional */, ResourceFormat::RGBA32Float},
    {"emissive", "gEmissive", "Emissive color", true /* optional */, ResourceFormat::RGBA32Float},
};

const char kSampleCount[] = "sampleCount";
const char kLightCount[] = "lightCount";

} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, LightSample>();
}

LightSample::LightSample(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mLightCount = 0;

    // Deserialize pass from dictionary.
    for (const auto& [key, value] : props)
    {
        if (key == kSampleCount)
            mSampleCount = (uint32_t)value;
        else if (key == kLightCount)
            mLightCount = (uint32_t)value;
        else
            logWarning("Unknown property '{}' in LightSample properties.", key);
    }
}

Properties LightSample::getProperties() const
{
    return {};
}

RenderPassReflection LightSample::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;

    // Add all outputs as UAVs. These are all optional.
    uint2 dim(mSampleCount, mLightCount);
    addRenderPassOutputs(reflector, kInputChannels, ResourceBindFlags::UnorderedAccess, dim);
    //printf("debug.LightSample.reflect dim=%u,%u\n", mSampleCount, mLightCount);

    return reflector;
}

void LightSample::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    // renderData holds the requested resources
    // auto& pTexture = renderData.getTexture("src");
    if (!mpSamplingPass)
    {
        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
        mpSamplingPass = ComputePass::create(mpDevice, kShaderFile, "main", defines);

        mpScene->setRaytracingShaderData(pRenderContext, mpSamplingPass->getRootVar());
    }

    auto var = mpSamplingPass->getRootVar();
    if (mpScene->getEnvMap() != nullptr)
    {
        if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
            mpEnvMapSampler = nullptr;

        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            mpEnvMapSampler->bindShaderData(var["envS"]);
        }
    }

    var["sceneRadius"] = mSceneRadius;

    var["gPosW"] = renderData.getTexture("posW");
    var["gNormW"] = renderData.getTexture("normW");
    var["gEmissive"] = renderData.getTexture("emissive");

    mpSamplingPass->execute(pRenderContext, uint3(mSampleCount, 1, 1));
}

void LightSample::renderUI(Gui::Widgets& widget)
{
    
}

void LightSample::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    mSceneRadius = pScene->getSceneBounds().radius();

    ref<const LightCollection> emissiveLight = pScene->getLightCollection(pRenderContext);
    uint32_t lightCount = emissiveLight->getMeshLights().size();

    if (pScene->getEnvMap() != nullptr)
        lightCount += 1u;

    mLightCount = lightCount;

    //printf("debug.LightSample.setScene mLightCount=%u\n", mLightCount);
}
