// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[ParseConfigDetPgie] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Engine output shape (with batch stripped): [84, 8400]
    // layer0.inferDims: numDims=2, d[0]=84, d[1]=8400
    // Layout: 84 rows x 8400 columns. Rows 0..3 = cx,cy,w,h; rows 4..83 = class scores.
    // Need to read in transposed order: for proposal j, feature i is at data[i * numProposals + j].

    const unsigned int featureDim = layer0.inferDims.d[0]; // 84
    const unsigned int numProposals = layer0.inferDims.d[1]; // 8400

    // Sanity: featureDim should be 4 + numClasses
    if (featureDim != 4u + numClasses) {
        fprintf(stderr, "[ParseConfigDetPgie] WARNING: featureDim=%u != 4+numClasses=%u\n",
                featureDim, 4u + numClasses);
    }

    // Only keep these COCO class ids (person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7)
    static const int allowedClasses[] = {0, 1, 2, 3, 5, 7};
    static const int numAllowed = 6;

    for (unsigned int j = 0; j < numProposals; ++j) {
        // Find best class score
        float maxScore = 0.0f;
        unsigned int bestClassId = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float s = data[(4u + c) * numProposals + j];
            if (s > maxScore) {
                maxScore = s;
                bestClassId = c;
            }
        }

        // Check if this class is in our allowed set
        bool allowed = false;
        for (int a = 0; a < numAllowed; ++a) {
            if (static_cast<int>(bestClassId) == allowedClasses[a]) {
                allowed = true;
                break;
            }
        }
        if (!allowed) continue;

        // Threshold
        float thr = (bestClassId < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClassId]
                     : 0.3f;
        if (maxScore < thr) continue;

        // Decode cx, cy, w, h (in network-input pixel space, 0..640)
        float cx = data[0u * numProposals + j];
        float cy = data[1u * numProposals + j];
        float bw = data[2u * numProposals + j];
        float bh = data[3u * numProposals + j];

        // Convert center to top-left
        float left = cx - bw * 0.5f;
        float top  = cy - bh * 0.5f;

        // Clamp to network input
        left = std::max(0.0f, std::min(left, netW));
        top  = std::max(0.0f, std::min(top,  netH));
        bw   = std::max(0.0f, std::min(bw, netW - left));
        bh   = std::max(0.0f, std::min(bh, netH - top));

        // Drop degenerate
        if (bw <= 1.0f || bh <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClassId;
        obj.left = left;
        obj.top = top;
        obj.width = bw;
        obj.height = bh;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}