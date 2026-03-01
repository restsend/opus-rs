/**
 * diag_silk_nlsf.c
 *
 * Diagnostic to capture SILK lpc_in_pre data from the C encoder.
 * Compiled with the opusic-sys source directly to access internal state.
 *
 * Build:
 *   cc -O0 -g -o /tmp/diag_silk_nlsf diag_silk_nlsf.c \
 *      -I/Users/pi/.cargo/registry/src/rsproxy.cn-e3de039b2554c837/opusic-sys-0.5.8/opus/include \
 *      -I/Users/pi/.cargo/registry/src/rsproxy.cn-e3de039b2554c837/opusic-sys-0.5.8/opus/silk \
 *      -I/Users/pi/.cargo/registry/src/rsproxy.cn-e3de039b2554c837/opusic-sys-0.5.8/opus/celt \
 *      -I/Users/pi/.cargo/registry/src/rsproxy.cn-e3de039b2554c837/opusic-sys-0.5.8/opus \
 *      /Users/pi/workspace/opus-rs/target/debug/libopus.0.dylib
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "opus.h"

int main(void) {
    const int SAMPLE_RATE = 8000;
    const int CHANNELS = 1;
    const int BITRATE = 10000;
    const int FRAME_MS = 20;
    const int FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS / 1000; /* 160 */
    
    int err;
    OpusEncoder *enc = opus_encoder_create(SAMPLE_RATE, CHANNELS,
                                           OPUS_APPLICATION_VOIP, &err);
    if (err != OPUS_OK) {
        fprintf(stderr, "Failed to create encoder: %d\n", err);
        return 1;
    }
    
    opus_encoder_ctl(enc, OPUS_SET_BITRATE(BITRATE));
    opus_encoder_ctl(enc, OPUS_SET_VBR(0));
    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(0));
    opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_NARROWBAND));
    opus_encoder_ctl(enc, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));
    
    /* Generate 440 Hz sine wave */
    float input[FRAME_SAMPLES];
    for (int i = 0; i < FRAME_SAMPLES; i++) {
        input[i] = sinf(2.0f * (float)M_PI * 440.0f * (float)i / (float)SAMPLE_RATE);
    }
    
    /* Print the raw input values converted to i16 (what C's FLOAT2INT16 gives) */
    printf("=== Raw input as i16 (first 20 samples) ===\n");
    for (int i = 0; i < 20; i++) {
        int val = (int)(0.5f + input[i] * 32768.0f);
        if (val > 32767) val = 32767;
        if (val < -32768) val = -32768;
        printf("  input_i16[%d] = %d\n", i, val);
    }
    
    /* Encode one frame - the C encoder will process it and print debug info */
    unsigned char packet[4096];
    int nbytes = opus_encode_float(enc, input, FRAME_SAMPLES, packet, sizeof(packet));
    if (nbytes < 0) {
        fprintf(stderr, "Encode failed: %d\n", nbytes);
        return 1;
    }
    
    printf("\nEncoded %d bytes\n", nbytes);
    printf("Hex: ");
    for (int i = 0; i < nbytes; i++) printf("%02x", packet[i]);
    printf("\n");
    
    opus_encoder_destroy(enc);
    return 0;
}
