use crate::range_coder::RangeCoder;
use crate::silk::decode_frame::{
    FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST, silk_decode_frame,
};
use crate::silk::decode_indices::{
    silk_decode_indices, silk_stereo_decode_mid_only, silk_stereo_decode_pred,
};
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::init_decoder::{silk_decoder_set_fs, silk_init_decoder, silk_reset_decoder};
use crate::silk::stereo_ms_to_lr::{StereoDecState, silk_stereo_ms_to_lr};
use crate::silk::tables::{SILK_LBRR_FLAGS_2_ICDF, SILK_LBRR_FLAGS_3_ICDF};

/// libopus `silk_DecControlStruct`: what the Opus layer tells SILK.
#[derive(Clone, Copy, Default)]
pub struct SilkDecControl {
    /// Output channels (1 or 2).
    pub n_channels_api: i32,
    /// Channels coded in the stream (1 or 2).
    pub n_channels_internal: i32,
    /// Output sampling rate in Hz.
    pub api_sample_rate: i32,
    /// SILK sampling rate in Hz (8000, 12000 or 16000).
    pub internal_sample_rate: i32,
    /// Duration of the packet's SILK payload: 10, 20, 40 or 60 ms.
    pub payload_size_ms: i32,
    /// Deep PLC conceals 16 kHz losses (decoder complexity 5 and above).
    #[cfg(feature = "deep-plc")]
    pub enable_deep_plc: bool,
    /// The SILK enhancer (libopus `osce_method`): LACE at decoder
    /// complexity 6, NoLACE from 7.
    #[cfg(feature = "osce")]
    pub osce_method: crate::dnn::osce::OsceMethod,
    /// The bandwidth extension is switched on (libopus `enable_osce_bwe`,
    /// `OPUS_SET_OSCE_BWE`).
    #[cfg(feature = "osce")]
    pub enable_osce_bwe: bool,
    /// libopus `osce_extended_mode`: `OSCE_MODE_SILK_BBWE` when BBWENet
    /// extends this frame to 48 kHz.
    #[cfg(feature = "osce")]
    pub osce_extended_mode: i32,
    /// libopus `prev_osce_extended_mode`.
    #[cfg(feature = "osce")]
    pub prev_osce_extended_mode: i32,
}

pub struct SilkDecoder {
    pub channel_state: [SilkDecoderState; 2],

    pub n_channels_api: i32,

    pub n_channels_internal: i32,

    pub prev_decode_only_middle: i32,

    /// Persistent stereo M/S decoder state.
    pub s_stereo: StereoDecState,

    /// Output rate of [`Self::decode`] and [`Self::decode_with_channels`],
    /// set by [`Self::init`].
    api_sample_rate: i32,

    /// Per-channel temp buffers for M/S → L/R conversion (frame_length + 2).
    w_silk_buf: [[i16; MAX_FRAME_LENGTH + 2]; 2],

    /// The deep PLC state of the Opus decoder (libopus `st->lpcnet`), kept
    /// here so that SILK's first channel and CELT can both reach it. It is
    /// created by [`Self::init_states`].
    #[cfg(feature = "deep-plc")]
    pub lpcnet: Option<crate::dnn::lpcnet_plc::LpcnetPlcState<'static>>,
    /// The OSCE models (libopus `silk_decoder.osce_model`).
    #[cfg(feature = "osce")]
    pub osce_model: Option<crate::dnn::osce::OsceModel<'static>>,
    /// The OSCE state of each channel (libopus `channel_state[n].osce`),
    /// created by [`Self::init_states`].
    #[cfg(feature = "osce")]
    pub osce: Option<[crate::dnn::osce::Osce; 2]>,
    /// The bandwidth extension state of each channel (libopus
    /// `channel_state[n].osce_bwe`).
    #[cfg(feature = "osce")]
    pub osce_bwe: Option<[crate::dnn::osce::OsceBwe; 2]>,
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SilkDecoder {
    /// A decoder for direct use: mono until [`Self::init`] says otherwise.
    pub fn new() -> Self {
        let mut decoder = Self::EMPTY;
        decoder.init_states();
        decoder.n_channels_api = 1;
        decoder.n_channels_internal = 1;
        decoder
    }

    /// A decoder for [`Self::init_states`] to set up. It is a constant so that
    /// it can be written straight into its final memory.
    pub(crate) const EMPTY: Self = Self {
        channel_state: [SilkDecoderState::DEFAULT, SilkDecoderState::DEFAULT],
        n_channels_api: 0,
        n_channels_internal: 0,
        prev_decode_only_middle: 0,
        s_stereo: StereoDecState::DEFAULT,
        api_sample_rate: 0,
        w_silk_buf: [[0; MAX_FRAME_LENGTH + 2]; 2],
        #[cfg(feature = "deep-plc")]
        lpcnet: None,
        #[cfg(feature = "osce")]
        osce_model: None,
        #[cfg(feature = "osce")]
        osce: None,
        #[cfg(feature = "osce")]
        osce_bwe: None,
    };

    /// libopus `silk_InitDecoder`.
    pub(crate) fn init_states(&mut self) {
        for state in &mut self.channel_state {
            silk_init_decoder(state);
        }
        self.s_stereo = StereoDecState::DEFAULT;
        self.prev_decode_only_middle = 0;
        self.n_channels_api = 0;
        self.n_channels_internal = 0;
        // libopus `lpcnet_plc_init` in opus_decoder_init.
        #[cfg(feature = "deep-plc")]
        {
            let models =
                crate::dnn::lpcnet_plc::PlcModels::builtin().expect("built-in deep PLC models");
            self.lpcnet = Some(crate::dnn::lpcnet_plc::LpcnetPlcState::new(models));
        }
        // silk_InitDecoder loads the built-in OSCE models; silk_init_decoder
        // resets each channel to OSCE_DEFAULT_METHOD.
        #[cfg(feature = "osce")]
        {
            self.osce_model = crate::dnn::osce::OsceModel::builtin().ok();
            self.osce = Some([crate::dnn::osce::Osce::new(), crate::dnn::osce::Osce::new()]);
            self.osce_bwe = Some([
                crate::dnn::osce::OsceBwe::new(),
                crate::dnn::osce::OsceBwe::new(),
            ]);
        }
    }

    /// Sets the decoder up for direct use with [`Self::decode`]: output at
    /// `sample_rate_hz` with `channels` channels, internal rate likewise.
    pub fn init(&mut self, sample_rate_hz: i32, channels: i32) -> i32 {
        let fs_khz = sample_rate_hz / 1000;
        self.api_sample_rate = sample_rate_hz;
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

    /// Decode one frame with the channel count used last time; see
    /// [`Self::decode_with_channels`]. Concealment keeps that count.
    pub fn decode(
        &mut self,
        range_dec: &mut RangeCoder,
        output: &mut [i16],
        lost_flag: i32,
        new_packet: bool,
        payload_size_ms: i32,
        internal_sample_rate: i32,
    ) -> i32 {
        self.decode_with_channels(
            range_dec,
            output,
            lost_flag,
            new_packet,
            payload_size_ms,
            internal_sample_rate,
            self.n_channels_internal,
        )
    }

    /// Decode one frame coded with `n_channels_internal` channels at the
    /// internal rate, without resampling. Output is planar, with
    /// `min(n_channels_api, n_channels_internal)` channels: a mono decoder
    /// gets the mid of a stereo stream, and a mono stream in a stereo decoder
    /// leaves the second channel to the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_with_channels(
        &mut self,
        range_dec: &mut RangeCoder,
        output: &mut [i16],
        lost_flag: i32,
        new_packet: bool,
        payload_size_ms: i32,
        internal_sample_rate: i32,
        n_channels_internal: i32,
    ) -> i32 {
        let ctrl = SilkDecControl {
            n_channels_api: self.n_channels_api,
            n_channels_internal,
            api_sample_rate: self.api_sample_rate,
            internal_sample_rate,
            payload_size_ms,
            ..SilkDecControl::default()
        };
        let (ret, fl, _) = self.decode_core(&ctrl, lost_flag, new_packet, range_dec);
        if ret < 0 {
            return -1;
        }
        let fl = fl as usize;
        if ctrl.n_channels_api == 2 && n_channels_internal == 2 {
            if output.len() < 2 * fl {
                return -1;
            }
            output[..fl].copy_from_slice(&self.w_silk_buf[0][1..1 + fl]);
            output[fl..2 * fl].copy_from_slice(&self.w_silk_buf[1][1..1 + fl]);
        } else {
            if output.len() < fl {
                return -1;
            }
            output[..fl].copy_from_slice(&self.w_silk_buf[0][1..1 + fl]);
        }
        fl as i32
    }

    /// libopus `silk_Decode` up to the stereo conversion: the frame lands at
    /// the internal rate in `w_silk_buf[n][1..=n_samples]`. Returns the error
    /// sum, the sample count and `stereo_to_mono`.
    fn decode_core(
        &mut self,
        ctrl: &SilkDecControl,
        lost_flag: i32,
        new_packet: bool,
        range_dec: &mut RangeCoder,
    ) -> (i32, i32, bool) {
        let mut ret = 0;
        let mut decode_only_middle = 0i32;
        let mut ms_pred_q13 = [0i32; 2];
        let n_internal = ctrl.n_channels_internal as usize;

        // Test if first frame in payload.
        if new_packet {
            for n in 0..n_internal {
                self.channel_state[n].n_frames_decoded = 0;
            }
        }

        // Mono -> stereo transition in the bitstream: init the second channel.
        if ctrl.n_channels_internal > self.n_channels_internal {
            ret += silk_init_decoder(&mut self.channel_state[1]);
            #[cfg(feature = "osce")]
            if let Some(states) = self.osce.as_mut() {
                crate::dnn::osce::osce_reset(&mut states[1], crate::dnn::osce::OSCE_DEFAULT_METHOD);
            }
            // silk_init_decoder resets the bandwidth extension too.
            #[cfg(feature = "osce")]
            if let Some(bwe) = self.osce_bwe.as_mut() {
                crate::dnn::osce::osce_bwe_reset(&mut bwe[1]);
            }
        }

        let stereo_to_mono = ctrl.n_channels_internal == 1
            && self.n_channels_internal == 2
            && ctrl.internal_sample_rate == 1000 * self.channel_state[0].fs_khz;

        if self.channel_state[0].n_frames_decoded == 0 {
            for n in 0..n_internal {
                let st = &mut self.channel_state[n];
                (st.n_frames_per_packet, st.nb_subfr) = match ctrl.payload_size_ms {
                    0 | 10 => (1, 2),
                    20 => (1, 4),
                    40 => (2, 4),
                    60 => (3, 4),
                    _ => return (-203, 0, stereo_to_mono),
                };
                let fs_khz_dec = (ctrl.internal_sample_rate >> 10) + 1;
                if fs_khz_dec != 8 && fs_khz_dec != 12 && fs_khz_dec != 16 {
                    return (-200, 0, stereo_to_mono);
                }
                ret += silk_decoder_set_fs(st, fs_khz_dec, ctrl.api_sample_rate);
            }
        }

        if ctrl.n_channels_api == 2
            && ctrl.n_channels_internal == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.s_stereo.pred_prev_q13 = [0; 2];
            self.s_stereo.s_side = [0; 2];
            self.channel_state[1].resampler_state = self.channel_state[0].resampler_state.clone();
        }
        self.n_channels_api = ctrl.n_channels_api;
        self.n_channels_internal = ctrl.n_channels_internal;

        if ctrl.api_sample_rate > 48000 || ctrl.api_sample_rate < 8000 {
            return (-200, 0, stereo_to_mono);
        }

        if lost_flag != FLAG_PACKET_LOST && self.channel_state[0].n_frames_decoded == 0 {
            // First decoder call for this payload: VAD flags and LBRR flag.
            for n in 0..n_internal {
                for i in 0..self.channel_state[n].n_frames_per_packet as usize {
                    self.channel_state[n].vad_flags[i] = range_dec.decode_bit_logp(1) as i32;
                }
                self.channel_state[n].lbrr_flag = range_dec.decode_bit_logp(1) as i32;
            }
            // LBRR flags.
            for n in 0..n_internal {
                let st = &mut self.channel_state[n];
                st.lbrr_flags.fill(0);
                if st.lbrr_flag != 0 {
                    if st.n_frames_per_packet == 1 {
                        st.lbrr_flags[0] = 1;
                    } else {
                        let icdf = if st.n_frames_per_packet == 2 {
                            &SILK_LBRR_FLAGS_2_ICDF[..]
                        } else {
                            &SILK_LBRR_FLAGS_3_ICDF[..]
                        };
                        let lbrr_symbol = range_dec.decode_icdf(icdf, 8) + 1;
                        for i in 0..st.n_frames_per_packet as usize {
                            st.lbrr_flags[i] = (lbrr_symbol >> i) & 1;
                        }
                    }
                }
            }

            if lost_flag == FLAG_DECODE_NORMAL {
                // Regular decoding: skip all LBRR data.
                for i in 0..self.channel_state[0].n_frames_per_packet as usize {
                    for n in 0..n_internal {
                        if self.channel_state[n].lbrr_flags[i] != 0 {
                            if n_internal == 2 && n == 0 {
                                ms_pred_q13 = silk_stereo_decode_pred(range_dec);
                                if self.channel_state[1].lbrr_flags[i] == 0 {
                                    decode_only_middle =
                                        silk_stereo_decode_mid_only(range_dec) as i32;
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

        // Get MS predictor index.
        if n_internal == 2 {
            let k = self.channel_state[0].n_frames_decoded as usize;
            if lost_flag == FLAG_DECODE_NORMAL
                || (lost_flag == FLAG_DECODE_LBRR && self.channel_state[0].lbrr_flags[k] == 1)
            {
                ms_pred_q13 = silk_stereo_decode_pred(range_dec);
                // For LBRR data, decode the mid-only flag only if the side's
                // LBRR flag is false.
                if (lost_flag == FLAG_DECODE_NORMAL && self.channel_state[1].vad_flags[k] == 0)
                    || (lost_flag == FLAG_DECODE_LBRR && self.channel_state[1].lbrr_flags[k] == 0)
                {
                    decode_only_middle = silk_stereo_decode_mid_only(range_dec) as i32;
                } else {
                    decode_only_middle = 0;
                }
            } else {
                ms_pred_q13 = self.s_stereo.pred_prev_q13;
            }
        }

        // Reset side channel decoder prediction memory for the first frame
        // with side coding.
        if n_internal == 2 && decode_only_middle == 0 && self.prev_decode_only_middle == 1 {
            let ch1 = &mut self.channel_state[1];
            ch1.out_buf.fill(0);
            ch1.s_lpc_q14_buf.fill(0);
            ch1.lag_prev = 100;
            ch1.last_gain_index = 10;
            ch1.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            ch1.first_frame_after_reset = 1;
        }

        let has_side = if lost_flag == FLAG_DECODE_NORMAL {
            decode_only_middle == 0
        } else {
            self.prev_decode_only_middle == 0
                || (n_internal == 2
                    && lost_flag == FLAG_DECODE_LBRR
                    && self.channel_state[1].lbrr_flags
                        [self.channel_state[1].n_frames_decoded as usize]
                        == 1)
        };

        #[cfg(feature = "deep-plc")]
        {
            self.channel_state[0].s_plc.enable_deep_plc = ctrl.enable_deep_plc as i32;
        }
        // Call the decoder for one frame.
        let mut n_samples_out_dec = 0i32;
        for n in 0..n_internal {
            if n == 0 || has_side {
                let frame_index = self.channel_state[0].n_frames_decoded - n as i32;
                // Use independent coding if no previous frame is available.
                let cond_coding = if frame_index <= 0 {
                    CODE_INDEPENDENTLY
                } else if lost_flag == FLAG_DECODE_LBRR {
                    if self.channel_state[n].lbrr_flags[frame_index as usize - 1] != 0 {
                        CODE_CONDITIONALLY
                    } else {
                        CODE_INDEPENDENTLY
                    }
                } else if n > 0 && self.prev_decode_only_middle != 0 {
                    // A side frame skipped in this packet needs no LTP
                    // scaling: the LTP state is well-defined.
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };
                #[cfg(feature = "deep-plc")]
                let lpcnet = if n == 0 { self.lpcnet.as_mut() } else { None };
                #[cfg(not(feature = "deep-plc"))]
                let lpcnet = None;
                #[cfg(feature = "osce")]
                let osce = self.osce.as_mut().map(|states| {
                    let state = &mut states[n];
                    if state.method != ctrl.osce_method {
                        crate::dnn::osce::osce_reset(state, ctrl.osce_method);
                    }
                    crate::silk::decode_frame::OsceArgs {
                        model: self.osce_model.as_ref(),
                        state,
                    }
                });
                #[cfg(not(feature = "osce"))]
                let osce = None;
                ret += silk_decode_frame(
                    &mut self.channel_state[n],
                    range_dec,
                    &mut self.w_silk_buf[n][2..],
                    &mut n_samples_out_dec,
                    lost_flag,
                    cond_coding,
                    lpcnet,
                    osce,
                );
            } else {
                let len = n_samples_out_dec as usize;
                self.w_silk_buf[n][2..2 + len].fill(0);
            }
            self.channel_state[n].n_frames_decoded += 1;
        }

        if ctrl.n_channels_api == 2 && n_internal == 2 {
            // Convert Mid/Side to Left/Right.
            let (buf_l, buf_r) = self.w_silk_buf.split_at_mut(1);
            silk_stereo_ms_to_lr(
                &mut self.s_stereo,
                &mut buf_l[0],
                &mut buf_r[0],
                &ms_pred_q13,
                self.channel_state[0].fs_khz,
                n_samples_out_dec,
            );
        } else {
            // Buffering.
            let n = n_samples_out_dec as usize;
            self.w_silk_buf[0][..2].copy_from_slice(&self.s_stereo.s_mid);
            self.s_stereo
                .s_mid
                .copy_from_slice(&self.w_silk_buf[0][n..n + 2]);
        }

        // Losses carry no mid-only flag (silk_decode_frame keeps the last
        // good one); concealment resets the gain clamping.
        if lost_flag == FLAG_PACKET_LOST {
            for n in 0..n_internal {
                self.channel_state[n].last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }

        (ret, n_samples_out_dec, stereo_to_mono)
    }

    /// libopus `silk_Decode`: decodes one SILK frame (10 or 20 ms) and
    /// resamples it to `ctrl.api_sample_rate`, interleaved for
    /// `ctrl.n_channels_api` channels into `samples_out`. Returns the error
    /// sum and the number of samples per channel written.
    pub fn silk_decode(
        &mut self,
        ctrl: &SilkDecControl,
        lost_flag: i32,
        new_packet: bool,
        range_dec: &mut RangeCoder,
        samples_out: &mut [f32],
    ) -> (i32, usize) {
        let (mut ret, n_dec, stereo_to_mono) =
            self.decode_core(ctrl, lost_flag, new_packet, range_dec);
        if n_dec <= 0 {
            return (if ret < 0 { ret } else { -1 }, 0);
        }
        let n_dec = n_dec as usize;
        let api_channels = ctrl.n_channels_api as usize;
        // Number of output samples.
        let n_out =
            n_dec * ctrl.api_sample_rate as usize / (self.channel_state[0].fs_khz as usize * 1000);
        let mut resampled = [0i16; 48 * MAX_FRAME_LENGTH_MS];
        let resample_out = &mut resampled[..n_out];

        let resample = |state: &mut SilkDecoderState, out: &mut [i16], inp: &[i16]| {
            state.resampler_state.process(out, inp, inp.len() as i32)
        };
        for n in 0..api_channels.min(ctrl.n_channels_internal as usize) {
            // Resample or extend the decoded signal to the API rate
            // (ENABLE_OSCE_BWE).
            #[cfg(feature = "osce")]
            if let (Some(model), Some(bwe)) = (self.osce_model.as_ref(), self.osce_bwe.as_mut()) {
                use crate::dnn::osce::{
                    OSCE_MODE_HYBRID, OSCE_MODE_SILK_BBWE, OSCE_MODE_SILK_ONLY,
                    features::osce_bwe_cross_fade_10ms, osce_bwe, osce_bwe_reset,
                };
                let input = &self.w_silk_buf[n][1..1 + n_dec];
                let mut resamp_buffer = [0i16; 3 * MAX_FRAME_LENGTH];
                if ctrl.osce_extended_mode == OSCE_MODE_SILK_BBWE {
                    if ctrl.prev_osce_extended_mode != OSCE_MODE_SILK_BBWE {
                        osce_bwe_reset(&mut bwe[n]);
                    }
                    osce_bwe(model, &mut bwe[n], resample_out, input);
                    if ctrl.prev_osce_extended_mode == OSCE_MODE_SILK_ONLY
                        || ctrl.prev_osce_extended_mode == OSCE_MODE_HYBRID
                    {
                        // Cross-fade with the upsampled signal.
                        resample(
                            &mut self.channel_state[n],
                            &mut resamp_buffer[..n_out],
                            input,
                        );
                        osce_bwe_cross_fade_10ms(resample_out, &resamp_buffer);
                    }
                } else {
                    ret += resample(&mut self.channel_state[n], resample_out, input);
                    if ctrl.prev_osce_extended_mode == OSCE_MODE_SILK_BBWE
                        && ctrl.internal_sample_rate == 16000
                    {
                        // Fade out if the internal sample rate did not change.
                        osce_bwe(model, &mut bwe[n], &mut resamp_buffer[..n_out], input);
                        osce_bwe_cross_fade_10ms(resample_out, &resamp_buffer);
                    }
                }
            } else {
                ret += resample(
                    &mut self.channel_state[n],
                    resample_out,
                    &self.w_silk_buf[n][1..1 + n_dec],
                );
            }
            #[cfg(not(feature = "osce"))]
            {
                ret += resample(
                    &mut self.channel_state[n],
                    resample_out,
                    &self.w_silk_buf[n][1..1 + n_dec],
                );
            }
            // Interleave if stereo output and stereo stream.
            for (i, &v) in resample_out.iter().enumerate() {
                samples_out[n + api_channels * i] = v as f32 * (1.0 / 32768.0);
            }
        }

        // Create two channel output from a mono stream.
        if api_channels == 2 && ctrl.n_channels_internal == 1 {
            if stereo_to_mono {
                // Resample the right channel for a newly collapsed stereo
                // stream, in case it was not collapsing when it switched.
                ret += resample(
                    &mut self.channel_state[1],
                    resample_out,
                    &self.w_silk_buf[0][1..1 + n_dec],
                );
                for (i, &v) in resample_out.iter().enumerate() {
                    samples_out[1 + 2 * i] = v as f32 * (1.0 / 32768.0);
                }
            } else {
                for i in 0..n_out {
                    samples_out[1 + 2 * i] = samples_out[2 * i];
                }
            }
        }

        (ret, n_out)
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

    /// libopus `silk_ResetDecoder`.
    pub fn reset(&mut self) {
        for state in &mut self.channel_state {
            silk_reset_decoder(state);
        }
        // silk_reset_decoder's osce_reset(OSCE_DEFAULT_METHOD) and
        // osce_bwe_reset.
        #[cfg(feature = "osce")]
        if let Some(states) = self.osce.as_mut() {
            for st in states {
                crate::dnn::osce::osce_reset(st, crate::dnn::osce::OSCE_DEFAULT_METHOD);
            }
        }
        #[cfg(feature = "osce")]
        if let Some(bwe) = self.osce_bwe.as_mut() {
            bwe.iter_mut().for_each(crate::dnn::osce::osce_bwe_reset);
        }
        self.s_stereo = StereoDecState::DEFAULT;
        self.prev_decode_only_middle = 0;
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

    /// Conceal one lost 20 ms stereo frame after a good frame whose mid-only
    /// flag was `prev_mid_only`, and return the decoder.
    fn conceal_after(prev_mid_only: i32) -> SilkDecoder {
        let mut dec = SilkDecoder::new();
        assert_eq!(dec.init(16000, 2), 0);
        dec.prev_decode_only_middle = prev_mid_only;
        let mut out = [0i16; 2 * 320];
        let mut rc = RangeCoder::new_decoder(&[]);
        let n = dec.decode(&mut rc, &mut out, FLAG_PACKET_LOST, true, 20, 16000);
        assert_eq!(n, 320);
        dec
    }

    /// libopus dec_API.c: a lost frame conceals the side channel only if
    /// the last good frame coded one, and leaves the mid-only flag alone.
    #[test]
    fn a_lost_frame_after_a_mid_only_frame_conceals_the_mid_only() {
        let dec = conceal_after(1);
        assert_eq!(dec.prev_decode_only_middle, 1);
        assert_eq!(dec.channel_state[0].loss_cnt, 1);
        assert_eq!(dec.channel_state[1].loss_cnt, 0, "side was concealed");
    }

    #[test]
    fn a_lost_frame_after_a_side_coded_frame_conceals_both() {
        let dec = conceal_after(0);
        assert_eq!(dec.prev_decode_only_middle, 0);
        assert_eq!(dec.channel_state[0].loss_cnt, 1);
        assert_eq!(dec.channel_state[1].loss_cnt, 1);
    }
}
