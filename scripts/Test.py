from falcor import *

def render_graph_TestPathTracer():
    g = RenderGraph("TestPathTracer")

    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1, 'maxSpecularBounces': 8})
    g.addPass(PathTracer, "PathTracer")

    # LightSample = createPass("LightSample", {'lightCount': 2, 'sampleCount': 256})
    # g.addPass(LightSample, "LightSample")

    GBufferRT = createPass("GBufferRT", {'maxBounces': 2, 'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(GBufferRT, "GBufferRT")

    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")

    AccumulatePrimaryPass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePrimaryPass, "AccumulatePrimaryPass")

    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("PathTracer.primary", "AccumulatePrimaryPass.input")
    g.markOutput("AccumulatePass.output")
    g.markOutput("AccumulatePrimaryPass.output")

    # g.markOutput("GBufferRT.posW")
    # g.markOutput("GBufferRT.normW")
    # g.markOutput("GBufferRT.refractPosW")
    # g.markOutput("GBufferRT.refractDirW")
    # g.markOutput("GBufferRT.finalPosW")
    g.markOutput("GBufferRT.diffRough")
    g.markOutput("GBufferRT.reflectPosW")
    g.markOutput("GBufferRT.reflectDirW")
    g.markOutput("GBufferRT.reflectDiffRough")
    g.markOutput("GBufferRT.reflectDepth")
    g.markOutput("GBufferRT.mask")
    # g.markOutput("GBufferRT.viewW")

    # g.markOutput("LightSample.posW")
    # g.markOutput("LightSample.normW")
    # g.markOutput("LightSample.emissive")

    return g

TestPathTracer = render_graph_TestPathTracer()
try: 
    builderFlags = SceneBuilderFlags.DontOptimizeGraph | SceneBuilderFlags.DontMergeMaterials
    m.loadScene(r'E:\workspace\tog\scenes\living-room\scene.pyscene', builderFlags)
    m.addGraph(TestPathTracer)
    # m.renderFrame()

    # print('debug begin')
    # transform = [
    #     1, 0, 0, 0.25,
    #     0, 1, 0, 1.25,
    #     0, 0, 1, 0.25,
    #     0, 0, 0, 1
    # ]
    # m.scene.update_node_matrix(7, transform)

except NameError: None
