#include "NeuralSurfaceSamplePass.h"
#include "Core/API/Formats.h"
#include "Utils/Logger.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralSurfaceSamplePass>();
}

NeuralSurfaceSamplePass::NeuralSurfaceSamplePass(ref<Device> device, const Properties& props) : RenderPass(device)
{
    for (const auto& [key, value] : props)
    {
        if (key == "sampleCount") mSampleCount = value;
        else if (key == "seed") mSeed = value;
        else logWarning("Unknown NeuralSurfaceSamplePass property '{}'", key);
    }
    mpCompute = ComputePass::create(device, "RenderPasses/NeuralSurfaceSamplePass/NeuralSurfaceSamplePass.cs.slang", "main");
}

Properties NeuralSurfaceSamplePass::getProperties() const
{
    Properties props;
    props["sampleCount"] = mSampleCount;
    props["seed"] = mSeed;
    return props;
}

RenderPassReflection NeuralSurfaceSamplePass::reflect(const CompileData& data)
{
    RenderPassReflection r;
    for (auto name : {"mask", "dirdepth", "position", "normal", "albedo", "metalrough", "emissive"})
    {
        r.addInput(name, name).texture2D(data.defaultTexDims.x, data.defaultTexDims.y);
        r.addOutput(std::string(name) + "Out", name)
            .bindFlags(ResourceBindFlags::UnorderedAccess)
            .format(ResourceFormat::RGBA32Float)
            .texture2D(mSampleCount, 1);
    }
    return r;
}

void NeuralSurfaceSamplePass::execute(RenderContext* context, const RenderData& data)
{
    auto var = mpCompute->getRootVar();
    var["CB"]["sampleCount"] = mSampleCount;
    var["CB"]["seed"] = mSeed;
    var["CB"]["sourceDim"] = uint2(data.getTexture("position")->getWidth(), data.getTexture("position")->getHeight());
    for (auto name : {"mask", "dirdepth", "position", "normal", "albedo", "metalrough", "emissive"})
    {
        var[std::string("g") + name] = data.getTexture(name);
        var[std::string("g") + name + "Out"] = data.getTexture(std::string(name) + "Out");
    }
    mpCompute->execute(context, uint3(mSampleCount, 1, 1));
}
