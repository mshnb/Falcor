from falcor import *

def render_graph_TestPathTracer():
    g = RenderGraph("TestPathTracer")

    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")

    LightSample = createPass("LightSample", {'lightCount': 2, 'sampleCount': 256})
    g.addPass(LightSample, "LightSample")

    GBufferRT = createPass("GBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")

    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")

    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.markOutput("AccumulatePass.output")

    g.markOutput("GBufferRT.posW")
    g.markOutput("GBufferRT.normW")

    g.markOutput("LightSample.posW")
    g.markOutput("LightSample.normW")
    g.markOutput("LightSample.emissive")

    return g

TestPathTracer = render_graph_TestPathTracer()
try: m.addGraph(TestPathTracer)
except NameError: None
