use crate::range_coder::RangeCoder;
use crate::silk::decode_frame::{FLAG_DECODE_NORMAL, FLAG_PACKET_LOST, silk_decode_frame};
use crate::silk::decode_indices::{
    silk_decode_indices, silk_stereo_decode_mid_only, silk_stereo_decode_pred,
};
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::init_decoder::{silk_decoder_set_fs, silk_init_decoder};
use crate::silk::stereo_ms_to_lr::{StereoDecState, silk_stereo_ms_to_lr};
use crate::silk::tables::{SILK_LBRR_FLAGS_2_ICDF, SILK_LBRR_FLAGS_3_ICDF};

pub struct SilkDecoder {
    pub channel_state: [SilkDecoderState; 2],

    pub n_channels_api: i32,

    pub n_channels_internal: i32,

    pub prev_decode_only_middle: i32,

    /// Persistent stereo M/S decoder state.
    pub s_stereo: StereoDecState,

    /// Per-channel temp buffers for M/S → L/R conversion (frame_length + 2).
    w_silk_buf: [[i16; MAX_FRAME_LENGTH + 2]; 2],
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SilkDecoder {
    pub fn new() -> Self {
        let mut dec = Self {
            channel_state: [SilkDecoderState::default(), SilkDecoderState::default()],
            n_channels_api: 1,
            n_channels_internal: 1,
            prev_decode_only_middle: 0,
            s_stereo: StereoDecState::default(),
            w_silk_buf: [[0; MAX_FRAME_LENGTH + 2]; 2],
        };
        silk_init_decoder(&mut dec.channel_state[0]);
        silk_init_decoder(&mut dec.channel_state[1]);
        dec
    }

    pub fn init(&mut self, sample_rate_hz: i32, channels: i32) -> i32 {
        let fs_khz = sample_rate_hz / 1000;
        let ret = silk_decoder_set_fs(&mut self.channel_state[0], fs_khz, sample_rate_hz);
        if ret < 0 {
            return ret;
        }
        if channels == 2 {
            let ret = silk_decoder_set_fs(&mut self.channel_state[1], fs_khz, sample_rate_hz);
            if ret < 0 {
                return ret;
            }
        }

        self.channel_state[0].n_frames_per_packet = 1;
        self.n_channels_api = channels;
        self.n_channels_internal = channels;
        ret
    }

    pub fn decode(
        &mut self,
        range_dec: &mut RangeCoder,
        output: &mut [i16],
        lost_flag: i32,
        new_packet: bool,
        payload_size_ms: i32,
        internal_sample_rate: i32,
    ) -> i32 {
        if new_packet {
            self.channel_state[0].n_frames_decoded = 0;
            self.channel_state[1].n_frames_decoded = 0;
        }

        if self.channel_state[0].n_frames_decoded == 0 {
            let n_channels = self.n_channels_internal as usize;
            for n in 0..n_channels {
                match payload_size_ms {
                    0 | 10 => {
                        self.channel_state[n].n_frames_per_packet = 1;
                        self.channel_state[n].nb_subfr = 2;
                    }
                    20 => {
                        self.channel_state[n].n_frames_per_packet = 1;
                        self.channel_state[n].nb_subfr = MAX_NB_SUBFR as i32;
                    }
                    40 => {
                        self.channel_state[n].n_frames_per_packet = 2;
                        self.channel_state[n].nb_subfr = MAX_NB_SUBFR as i32;
                    }
                    60 => {
                        self.channel_state[n].n_frames_per_packet = 3;
                        self.channel_state[n].nb_subfr = MAX_NB_SUBFR as i32;
                    }
                    _ => return -1,
                }
            }

            let fs_khz_dec = (internal_sample_rate >> 10) + 1;
            if fs_khz_dec != 8 && fs_khz_dec != 12 && fs_khz_dec != 16 {
                return -1;
            }
            let api_sample_rate = self.channel_state[0].fs_api_hz;
            for n in 0..n_channels {
                let ret =
                    silk_decoder_set_fs(&mut self.channel_state[n], fs_khz_dec, api_sample_rate);
                if ret < 0 {
                    return ret;
                }
                if payload_size_ms == 10 {
                    self.channel_state[n].nb_subfr = 2;
                    self.channel_state[n].frame_length = self.channel_state[n].subfr_length * 2;
                }
            }
        }

        // Initialise second channel and stereo state on mono→stereo transition.
        if self.n_channels_internal == 2
            && self.prev_decode_only_middle == 0
            && self.s_stereo.pred_prev_q13 == [0, 0]
        {
            // Fresh stereo init: clear stereo state.
            self.s_stereo = StereoDecState::default();
        }

        if lost_flag != FLAG_PACKET_LOST && self.channel_state[0].n_frames_decoded == 0 {
            let n_frames_per_packet = self.channel_state[0].n_frames_per_packet.max(1);
            let n_channels = self.n_channels_internal as usize;

            for n in 0..n_channels {
                for i in 0..n_frames_per_packet as usize {
                    let vad = range_dec.decode_bit_logp(1);
                    self.channel_state[n].vad_flags[i] = if vad { 1 } else { 0 };
                }
                let lbrr = range_dec.decode_bit_logp(1);
                self.channel_state[n].lbrr_flag = if lbrr { 1 } else { 0 };
            }

            for n in 0..n_channels {
                self.channel_state[n].lbrr_flags.fill(0);
                if self.channel_state[n].lbrr_flag != 0 {
                    if n_frames_per_packet == 1 {
                        self.channel_state[n].lbrr_flags[0] = 1;
                    } else {
                        let lbrr_icdf = match n_frames_per_packet {
                            2 => &SILK_LBRR_FLAGS_2_ICDF[..],
                            3 => &SILK_LBRR_FLAGS_3_ICDF[..],
                            _ => &SILK_LBRR_FLAGS_2_ICDF[..],
                        };
                        let lbrr_symbol = range_dec.decode_icdf(lbrr_icdf, 8) + 1;
                        for i in 0..n_frames_per_packet as usize {
                            self.channel_state[n].lbrr_flags[i] = (lbrr_symbol >> i) & 1;
                        }
                    }
                }
            }

            // Skip LBRR data for all frames/channels
            if lost_flag == FLAG_DECODE_NORMAL {
                for i in 0..n_frames_per_packet as usize {
                    for n in 0..n_channels {
                        if self.channel_state[n].lbrr_flags[i] != 0 {
                            if n_channels == 2 && n == 0 {
                                let _ = silk_stereo_decode_pred(range_dec);
                                if self.channel_state[1].lbrr_flags[i] == 0 {
                                    let _ = silk_stereo_decode_mid_only(range_dec);
                                }
                            }
                            let cond_coding =
                                if i > 0 && self.channel_state[n].lbrr_flags[i - 1] != 0 {
                                    CODE_CONDITIONALLY
                                } else {
                                    CODE_INDEPENDENTLY
                                };
                            silk_decode_indices(
                                &mut self.channel_state[n],
                                range_dec,
                                i as i32,
                                1,
                                cond_coding,
                            );
                            let mut pulses = [0i16; MAX_FRAME_LENGTH];
                            silk_decode_pulses(
                                range_dec,
                                &mut pulses,
                                self.channel_state[n].indices.signal_type as i32,
                                self.channel_state[n].indices.quant_offset_type as i32,
                                self.channel_state[n].frame_length,
                            );
                        }
                    }
                }
            }
        }

        // Decode M/S predictors for stereo.
        let mut ms_pred_q13 = [0i32; 2];
        let mut decode_only_middle = 0i32;
        let frame_index = self.channel_state[0].n_frames_decoded as usize;
        if self.n_channels_internal == 2 {
            if lost_flag == FLAG_DECODE_NORMAL {
                ms_pred_q13 = silk_stereo_decode_pred(range_dec);
                if self.channel_state[1].vad_flags[frame_index] == 0 {
                    decode_only_middle = if silk_stereo_decode_mid_only(range_dec) {
                        1
                    } else {
                        0
                    };
                }
            } else {
                ms_pred_q13 = self.s_stereo.pred_prev_q13;
            }
        }

        // Reset side-channel prediction memory when transitioning from mid-only to has-side.
        if self.n_channels_internal == 2
            && decode_only_middle == 0
            && self.prev_decode_only_middle == 1
        {
            let ch1 = &mut self.channel_state[1];
            ch1.out_buf.fill(0);
            ch1.s_lpc_q14_buf.fill(0);
            ch1.lag_prev = 100;
            ch1.last_gain_index = 10;
            ch1.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            ch1.first_frame_after_reset = 1;
        }

        let has_side = decode_only_middle == 0;
        let n_channels = self.n_channels_internal as usize;
        let frame_length = self.channel_state[0].frame_length as usize;

        // Decode each channel into w_silk_buf[n][2..2+frame_length].
        let mut n_samples_out: i32 = 0;
        for n in 0..n_channels {
            if n == 0 || has_side {
                let frame_idx = self.channel_state[0].n_frames_decoded - n as i32;
                let cond_coding = if frame_idx <= 0 {
                    CODE_INDEPENDENTLY
                } else if n > 0 && self.prev_decode_only_middle != 0 {
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };

                // Clear the overlap region before decode.
                self.w_silk_buf[n][0] = 0;
                self.w_silk_buf[n][1] = 0;

                let mut ns = 0i32;
                let ret = silk_decode_frame(
                    &mut self.channel_state[n],
                    range_dec,
                    &mut self.w_silk_buf[n][2..],
                    &mut ns,
                    lost_flag,
                    cond_coding,
                );
                if ret < 0 {
                    return ret;
                }
                if n == 0 {
                    n_samples_out = ns;
                }
            } else {
                // Mid-only: zero the side channel.
                for i in 0..frame_length {
                    self.w_silk_buf[n][2 + i] = 0;
                }
            }
            self.channel_state[n].n_frames_decoded += 1;
        }

        // Convert Mid/Side → Left/Right for stereo, or do mono buffering.
        if n_channels == 2 {
            let (buf_l, buf_r) = self.w_silk_buf.split_at_mut(1);
            silk_stereo_ms_to_lr(
                &mut self.s_stereo,
                &mut buf_l[0],
                &mut buf_r[0],
                &ms_pred_q13,
                self.channel_state[0].fs_khz,
                n_samples_out,
            );
        } else {
            // Mono buffering: save overlap for next frame.
            self.w_silk_buf[0][0] = self.s_stereo.s_mid[0];
            self.w_silk_buf[0][1] = self.s_stereo.s_mid[1];
            self.s_stereo.s_mid[0] = self.w_silk_buf[0][n_samples_out as usize];
            self.s_stereo.s_mid[1] = self.w_silk_buf[0][n_samples_out as usize + 1];
        }

        // Copy planar output: ch0 at output[0..fl], ch1 at output[fl..2*fl].
        // For stereo, MS_to_LR outputs at [1..1+fl] (with overlap compensation).
        // For mono, decoded data is at [2..2+fl] — use it directly (no offset)
        // to match the pre-stereo-rewrite behaviour.
        let fl = n_samples_out as usize;
        if n_channels == 2 {
            if output.len() < 2 * fl {
                return -1;
            }
            output[..fl].copy_from_slice(&self.w_silk_buf[0][1..1 + fl]);
            output[fl..2 * fl].copy_from_slice(&self.w_silk_buf[1][1..1 + fl]);
        } else {
            if output.len() < fl {
                return -1;
            }
            output[..fl].copy_from_slice(&self.w_silk_buf[0][2..2 + fl]);
        }

        self.prev_decode_only_middle = decode_only_middle;

        if n_samples_out < 0 { -1 } else { n_samples_out }
    }

    pub fn decode_bytes(&mut self, data: &[u8], output: &mut [i16], new_packet: bool) -> i32 {
        let mut range_dec = RangeCoder::new_decoder(data);
        let internal_rate = self.channel_state[0].fs_khz * 1000;
        let payload_ms = if self.channel_state[0].nb_subfr == 2 {
            10
        } else {
            20
        };
        self.decode(
            &mut range_dec,
            output,
            FLAG_DECODE_NORMAL,
            new_packet,
            payload_ms,
            internal_rate,
        )
    }

    pub fn reset(&mut self) {
        silk_init_decoder(&mut self.channel_state[0]);
        silk_init_decoder(&mut self.channel_state[1]);
        self.prev_decode_only_middle = 0;
        self.s_stereo = StereoDecState::default();
    }

    pub fn frame_length(&self) -> i32 {
        self.channel_state[0].frame_length
    }

    pub fn sample_rate(&self) -> i32 {
        self.channel_state[0].fs_khz * 1000
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let dec = SilkDecoder::new();
        assert_eq!(dec.n_channels_api, 1);
        assert_eq!(dec.n_channels_internal, 1);
    }

    #[test]
    fn test_decoder_init() {
        let mut dec = SilkDecoder::new();

        let ret = dec.init(16000, 1);
        assert_eq!(ret, 0);
        assert_eq!(dec.sample_rate(), 16000);
    }

    #[test]
    fn test_decoder_16khz() {
        let mut dec = SilkDecoder::new();
        let ret = dec.init(16000, 1);
        assert_eq!(ret, 0);
        assert_eq!(dec.frame_length(), 320);
    }
}
