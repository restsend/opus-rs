/* Records the samples opus_encoder.c hands to silk_Encode (see trace.c). */
#ifndef OFZ_SILK_HOOK_H
#define OFZ_SILK_HOOK_H
#include "API.h"
opus_int ofz_silk_Encode(void *enc, silk_EncControlStruct *ctl, const opus_res *in, opus_int n,
                         ec_enc *rc, opus_int32 *bytes, const opus_int prefill, int activity);
#define silk_Encode ofz_silk_Encode
#endif
