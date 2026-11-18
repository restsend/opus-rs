/* Creates and runs encoders and decoders on a thread with a 128 KiB
   stack: the C library builds its states in place, without a stack
   temporary of them. */
#include <opus.h>
#include <opus_multistream.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "failed: %s\n", #x); exit(1); } } while (0)

static void *run(void *arg)
{
   static float pcm[960 * 6];
   static float out[960 * 6];
   unsigned char packet[4000];
   int err, len, streams, coupled;
   unsigned char mapping[6];
   OpusEncoder *enc;
   OpusDecoder *dec;
   OpusMSEncoder *msenc;
   OpusMSDecoder *msdec;
   (void)arg;

   enc = opus_encoder_create(48000, 2, OPUS_APPLICATION_AUDIO, &err);
   CHECK(err == OPUS_OK);
   len = opus_encode_float(enc, pcm, 960, packet, sizeof(packet));
   CHECK(len > 0);
   dec = opus_decoder_create(48000, 2, &err);
   CHECK(err == OPUS_OK);
   CHECK(opus_decode_float(dec, packet, len, out, 960, 0) == 960);
   opus_encoder_destroy(enc);
   opus_decoder_destroy(dec);

   msenc = opus_multistream_surround_encoder_create(48000, 6, 1, &streams, &coupled, mapping,
      OPUS_APPLICATION_AUDIO, &err);
   CHECK(err == OPUS_OK);
   len = opus_multistream_encode_float(msenc, pcm, 960, packet, sizeof(packet));
   CHECK(len > 0);
   msdec = opus_multistream_decoder_create(48000, 6, streams, coupled, mapping, &err);
   CHECK(err == OPUS_OK);
   CHECK(opus_multistream_decode_float(msdec, packet, len, out, 960, 0) == 960);
   opus_multistream_encoder_destroy(msenc);
   opus_multistream_decoder_destroy(msdec);
   return NULL;
}

int main(void)
{
   pthread_attr_t attr;
   pthread_t thread;
   CHECK(pthread_attr_init(&attr) == 0);
   CHECK(pthread_attr_setstacksize(&attr, 128 * 1024) == 0);
   CHECK(pthread_create(&thread, &attr, run, NULL) == 0);
   CHECK(pthread_join(thread, NULL) == 0);
   printf("Encoders and decoders ran on a 128 KiB stack.\n");
   return 0;
}
