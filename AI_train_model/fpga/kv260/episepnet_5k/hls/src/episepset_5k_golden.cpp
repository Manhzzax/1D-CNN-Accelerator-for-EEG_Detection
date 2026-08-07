#include "episepset_5k_contract.h"
#include "episepset_5k_weights.h"

#include <cfenv>
#include <cmath>
#include <cstdint>

namespace episepset_5k {
namespace {

std::int16_t saturate_int16(std::int64_t value) {
    if (value > kInt16Limit) {
        return static_cast<std::int16_t>(kInt16Limit);
    }
    if (value < -kInt16Limit) {
        return static_cast<std::int16_t>(-kInt16Limit);
    }
    return static_cast<std::int16_t>(value);
}

std::int16_t requantize_m1a(std::int64_t accumulator, double ratio, bool relu) {
    const auto rounded = static_cast<std::int64_t>(std::nearbyint(static_cast<double>(accumulator) * ratio));
    const auto saturated = saturate_int16(rounded);
    return relu && saturated < 0 ? 0 : saturated;
}

std::int16_t average_pool4(const std::int16_t values[4]) {
    const std::int64_t sum = static_cast<std::int64_t>(values[0]) + values[1] + values[2] + values[3];
    return static_cast<std::int16_t>((sum + 2) / 4);
}

}  // namespace

void episepset_5k_golden(
    const std::int16_t input[kChannels][kSamples],
    std::int64_t logits[2]
) {
    std::fesetround(FE_TONEAREST);
    std::int16_t temporal[kTemporalChannels][kSamples] = {};
    std::int16_t spatial[kSpatialChannels][kSamples] = {};
    std::int16_t pooled_spatial[kSpatialChannels][kPooledSamples] = {};
    std::int16_t refine_depthwise[kSpatialChannels][kPooledSamples] = {};
    std::int16_t refine[kSpatialChannels][kPooledSamples] = {};
    std::int16_t pooled_refine[kSpatialChannels][kFinalSamples] = {};
    std::int16_t global[kSpatialChannels] = {};

    for (int output_channel = 0; output_channel < kTemporalChannels; ++output_channel) {
        const int input_channel = output_channel / 3;
        for (int time = 0; time < kSamples; ++time) {
            std::int64_t accumulator = temporal_depthwise_bias[output_channel];
            for (int kernel = 0; kernel < kTemporalKernel; ++kernel) {
                const int source_time = time + kernel - (kTemporalKernel / 2);
                if (source_time >= 0 && source_time < kSamples) {
                    accumulator += static_cast<std::int64_t>(input[input_channel][source_time]) * temporal_depthwise_weight[output_channel][kernel];
                }
            }
            temporal[output_channel][time] = requantize_m1a(accumulator, kTemporalRequantRatio, true);
        }
    }

    for (int output_channel = 0; output_channel < kSpatialChannels; ++output_channel) {
        for (int time = 0; time < kSamples; ++time) {
            std::int64_t accumulator = spatial_pointwise_bias[output_channel];
            for (int input_channel = 0; input_channel < kTemporalChannels; ++input_channel) {
                accumulator += static_cast<std::int64_t>(temporal[input_channel][time]) * spatial_pointwise_weight[output_channel][input_channel];
            }
            spatial[output_channel][time] = requantize_m1a(accumulator, kSpatialRequantRatio, true);
        }
    }

    for (int channel = 0; channel < kSpatialChannels; ++channel) {
        for (int output_time = 0; output_time < kPooledSamples; ++output_time) {
            pooled_spatial[channel][output_time] = average_pool4(&spatial[channel][output_time * kPool]);
        }
    }

    for (int channel = 0; channel < kSpatialChannels; ++channel) {
        for (int time = 0; time < kPooledSamples; ++time) {
            std::int64_t accumulator = 0;
            for (int kernel = 0; kernel < kRefinementKernel; ++kernel) {
                const int source_time = time + kernel - (kRefinementKernel / 2);
                if (source_time >= 0 && source_time < kPooledSamples) {
                    accumulator += static_cast<std::int64_t>(pooled_spatial[channel][source_time]) * refine_depthwise_weight[channel][kernel];
                }
            }
            refine_depthwise[channel][time] = requantize_m1a(accumulator, kRefineDepthwiseRequantRatio, false);
        }
    }

    for (int output_channel = 0; output_channel < kSpatialChannels; ++output_channel) {
        for (int time = 0; time < kPooledSamples; ++time) {
            std::int64_t accumulator = refine_pointwise_bias[output_channel];
            for (int input_channel = 0; input_channel < kSpatialChannels; ++input_channel) {
                accumulator += static_cast<std::int64_t>(refine_depthwise[input_channel][time]) * refine_pointwise_weight[output_channel][input_channel];
            }
            refine[output_channel][time] = requantize_m1a(accumulator, kRefinePointwiseRequantRatio, true);
        }
    }

    for (int channel = 0; channel < kSpatialChannels; ++channel) {
        for (int output_time = 0; output_time < kFinalSamples; ++output_time) {
            pooled_refine[channel][output_time] = average_pool4(&refine[channel][output_time * kPool]);
        }
        std::int64_t sum = 0;
        for (int time = 0; time < kFinalSamples; ++time) {
            sum += pooled_refine[channel][time];
        }
        global[channel] = static_cast<std::int16_t>((sum + (kFinalSamples / 2)) / kFinalSamples);
    }

    for (int output = 0; output < 2; ++output) {
        std::int64_t accumulator = classifier_bias[output];
        for (int channel = 0; channel < kSpatialChannels; ++channel) {
            accumulator += static_cast<std::int64_t>(global[channel]) * classifier_weight[output][channel];
        }
        logits[output] = accumulator;
    }
}

}  // namespace episepset_5k
