from falcor import *

def render_graph_TestPathTracer():
    g = RenderGraph("TestPathTracer")

    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")

    # LightSample = createPass("LightSample", {'lightCount': 2, 'sampleCount': 256})
    # g.addPass(LightSample, "LightSample")

    GBufferRT = createPass("GBufferRT", {'maxBounces': 1, 'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")

    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")

    DirectAccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(DirectAccumulatePass, "DirectAccumulatePass")

    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("PathTracer.direct", "DirectAccumulatePass.input")
    g.markOutput("AccumulatePass.output")
    g.markOutput("DirectAccumulatePass.output")

    # g.markOutput("GBufferRT.posW")
    # g.markOutput("GBufferRT.normW")
    # g.markOutput("GBufferRT.refractPosW")
    # g.markOutput("GBufferRT.refractDirW")
    # g.markOutput("GBufferRT.finalPosW")
    # g.markOutput("GBufferRT.mask")
    # g.markOutput("GBufferRT.viewW")

    # g.markOutput("LightSample.posW")
    # g.markOutput("LightSample.normW")
    # g.markOutput("LightSample.emissive")

    return g

TestPathTracer = render_graph_TestPathTracer()
try: 
    builderFlags = SceneBuilderFlags.DontOptimizeGraph | SceneBuilderFlags.DontMergeMaterials
    m.loadScene(r'E:\workspace\tog\neural_dielectric\scenes\cornell-glossy\scene.pyscene', builderFlags)
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
