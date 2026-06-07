#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Persistent-kernel launcher wrapper generated alongside PTX.
void qfx_launch(const float *prices, const float *volumes, float *out0, int n, int w,
                void *stream);

#ifdef __cplusplus
}
#endif
