/**
 * verify_silk_encode.c
 *
 * Dumps internal SILK encoder parameters from the C reference (libopus)
 * for comparison with the Rust implementation.
 *
 * Build: cc -O2 -o /tmp/verify_silk_encode verify_silk_encode.c \
 *        -I opus-1.6/include -L opus-1.6/.libs -lopus -lm \
 *        -Wl,-rpath,opus-1.6/.libs
 * Run: /tmp/verify_silk_encode
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "opus.h"

int main(void)
{
    /* Match the bitstream_compare test: 8kHz NB, mono, 10kbps CBR, complexity 0 */
    const int SAMPLE_RATE = 8000;
    const int CHANNELS    = 1;
    const int BITRATE     = 10000;
    const int FRAME_MS    = 20;
    const int FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS / 1000; /* 160 */
    const int MAX_PACKET  = 1275;

    int err;
    OpusEncoder *enc = opus_encoder_create(SAMPLE_RATE, CHANNELS,
                                           OPUS_APPLICATION_VOIP, &err);
    if (err != OPUS_OK) {
        fprintf(stderr, "opus_encoder_create failed: %d\n", err);
        return 1;
    }

    opus_encoder_ctl(enc, OPUS_SET_BITRATE(BITRATE));
    opus_encoder_ctl(enc, OPUS_SET_VBR(0));            /* CBR */
    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(0));
    opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_NARROWBAND));
    opus_encoder_ctl(enc, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));

    /* Generate 440 Hz sine wave (matches Rust test_silk_bitstream_vs_c_reference) */
    float input[FRAME_SAMPLES];
    for (int i = 0; i < FRAME_SAMPLES; i++) {
        input[i] = sinf(2.0f * (float)M_PI * 440.0f * (float)i / (float)SAMPLE_RATE);
    }

    /* Encode 3 frames */
    unsigned char packet[MAX_PACKET];
    for (int frame = 0; frame < 3; frame++) {
        int nbytes = opus_encode_float(enc, input, FRAME_SAMPLES, packet, MAX_PACKET);
        if (nbytes < 0) {
            fprintf(stderr, "opus_encode_float failed: %d\n", nbytes);
            return 1;
        }

        printf("Frame %d: %d bytes\n", frame, nbytes);
        printf("  Hex: ");
        for (int i = 0; i < nbytes; i++) printf("%02x", packet[i]);
        printf("\n");

        /* Parse TOC */
        int toc = packet[0];
        int config = (toc >> 3) & 0x1F;
        int stereo = (toc >> 2) & 1;
        int code   = toc & 3;
        printf("  TOC: config=%d stereo=%d code=%d\n", config, stereo, code);
    }

    /* Also encode a constant-amplitude sine for comparison */
    printf("\n--- 16kHz WB test ---\n");
    OpusEncoder *enc2 = opus_encoder_create(16000, 1, OPUS_APPLICATION_VOIP, &err);
    opus_encoder_ctl(enc2, OPUS_SET_BITRATE(BITRATE));
    opus_encoder_ctl(enc2, OPUS_SET_VBR(0));
    opus_encoder_ctl(enc2, OPUS_SET_COMPLEXITY(1));
    opus_encoder_ctl(enc2, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_WIDEBAND));
    opus_encoder_ctl(enc2, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));

    int frame_samples_16k = 16000 * FRAME_MS / 1000; /* 320 */
    float *input16k = calloc(frame_samples_16k, sizeof(float));
    for (int i = 0; i < frame_samples_16k; i++) {
        input16k[i] = 0.5f * sinf(2.0f * (float)M_PI * 200.0f * (float)i / 16000.0f);
    }

    for (int frame = 0; frame < 3; frame++) {
        int nbytes = opus_encode_float(enc2, input16k, frame_samples_16k, packet, MAX_PACKET);
        printf("Frame %d: %d bytes\n", frame, nbytes);
        printf("  Hex: ");
        for (int i = 0; i < nbytes; i++) printf("%02x", packet[i]);
        printf("\n");
    }

    free(input16k);
    opus_encoder_destroy(enc);
    opus_encoder_destroy(enc2);
    return 0;
}
