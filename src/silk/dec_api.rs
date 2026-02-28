use crate::range_coder::RangeCoder;
use crate::silk::decode_frame::{FLAG_DECODE_NORMAL, FLAG_PACKET_LOST, silk_decode_frame};
use crate::silk::decode_indices::silk_decode_indices;
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::init_decoder::{silk_decoder_set_fs, silk_init_decoder};
use crate::silk::tables::{SILK_LBRR_FLAGS_2_ICDF, SILK_LBRR_FLAGS_3_ICDF};

/// SILK decoder wrapper
pub struct SilkDecoder {
    /// Channel states (for stereo support)
    pub channel_state: [SilkDecoderState; 2],
    /// Number of channels in API
    pub n_channels_api: i32,
    /// Number of internal channels
    pub n_channels_internal: i32,
    /// Previous decode only middle flag (for stereo)
    pub prev_decode_only_middle: i32,
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SilkDecoder {
    /// Create a new SILK decoder
    pub fn new() -> Self {
        let mut dec = Self {
            channel_state: [SilkDecoderState::default(), SilkDecoderState::default()],
            n_channels_api: 1,
            n_channels_internal: 1,
            prev_decode_only_middle: 0,
        };
        silk_init_decoder(&mut dec.channel_state[0]);
        silk_init_decoder(&mut dec.channel_state[1]);
        dec
    }

    /// Initialize the decoder for a specific sample rate
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
        // Set n_frames_per_packet based on frame length
        // For 20ms frames at the given sample rate
        self.channel_state[0].n_frames_per_packet = 1; // Assume 1 frame per packet for now
        self.n_channels_api = channels;
        self.n_channels_internal = channels;
        ret
    }

    /// Decode a SILK frame from range coder data
    ///
    /// Matches C `silk_Decode()` in `dec_API.c`.
    /// `payload_size_ms`: frame duration in ms (10/20/40/60), used on first call
    /// `internal_sample_rate`: internal SILK sample rate (8000/12000/16000)
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

        // First frame: set up nFramesPerPacket and nb_subfr from payloadSize_ms
        if self.channel_state[0].n_frames_decoded == 0 {
            match payload_size_ms {
                0 | 10 => {
                    self.channel_state[0].n_frames_per_packet = 1;
                    self.channel_state[0].nb_subfr = 2;
                }
                20 => {
                    self.channel_state[0].n_frames_per_packet = 1;
                    self.channel_state[0].nb_subfr = MAX_NB_SUBFR as i32;
                }
                40 => {
                    self.channel_state[0].n_frames_per_packet = 2;
                    self.channel_state[0].nb_subfr = MAX_NB_SUBFR as i32;
                }
                60 => {
                    self.channel_state[0].n_frames_per_packet = 3;
                    self.channel_state[0].nb_subfr = MAX_NB_SUBFR as i32;
                }
                _ => return -1, // SILK_DEC_INVALID_FRAME_SIZE
            }

            // Compute fs_kHz and call decoder_set_fs
            let fs_khz_dec = (internal_sample_rate >> 10) + 1;
            if fs_khz_dec != 8 && fs_khz_dec != 12 && fs_khz_dec != 16 {
                return -1; // SILK_DEC_INVALID_SAMPLING_FREQUENCY
            }
            let api_sample_rate = self.channel_state[0].fs_api_hz;
            let ret = silk_decoder_set_fs(&mut self.channel_state[0], fs_khz_dec, api_sample_rate);
            if ret < 0 {
                return ret;
            }
        }

        // Decode VAD flags and LBRR flags before decoding the first frame
        if lost_flag != FLAG_PACKET_LOST && self.channel_state[0].n_frames_decoded == 0 {
            let n_frames_per_packet = self.channel_state[0].n_frames_per_packet.max(1);

            // Decode VAD flags
            for i in 0..n_frames_per_packet as usize {
                let vad = range_dec.decode_bit_logp(1);
                self.channel_state[0].vad_flags[i] = if vad { 1 } else { 0 };
            }
            // Decode LBRR flag
            let lbrr = range_dec.decode_bit_logp(1);
            self.channel_state[0].lbrr_flag = if lbrr { 1 } else { 0 };

            // Decode LBRR sub-flags
            self.channel_state[0].lbrr_flags.fill(0);
            if self.channel_state[0].lbrr_flag != 0 {
                if n_frames_per_packet == 1 {
                    self.channel_state[0].lbrr_flags[0] = 1;
                } else {
                    // C: LBRR_symbol = ec_dec_icdf(psRangeDec, silk_LBRR_flags_iCDF_ptr[nFramesPerPacket - 2], 8) + 1;
                    let lbrr_icdf = match n_frames_per_packet {
                        2 => &SILK_LBRR_FLAGS_2_ICDF[..],
                        3 => &SILK_LBRR_FLAGS_3_ICDF[..],
                        _ => &SILK_LBRR_FLAGS_2_ICDF[..],
                    };
                    let lbrr_symbol = range_dec.decode_icdf(lbrr_icdf, 8) + 1;
                    for i in 0..n_frames_per_packet as usize {
                        self.channel_state[0].lbrr_flags[i] = (lbrr_symbol >> i) & 1;
                    }
                }
            }

            // For normal decoding: skip all LBRR data in the bitstream
            if lost_flag == FLAG_DECODE_NORMAL {
                for i in 0..n_frames_per_packet as usize {
                    if self.channel_state[0].lbrr_flags[i] != 0 {
                        // Use conditional coding if previous LBRR frame available
                        let cond_coding = if i > 0 && self.channel_state[0].lbrr_flags[i - 1] != 0 {
                            CODE_CONDITIONALLY
                        } else {
                            CODE_INDEPENDENTLY
                        };
                        // Decode indices (consume bitstream, state updated for LBRR)
                        silk_decode_indices(
                            &mut self.channel_state[0],
                            range_dec,
                            i as i32,
                            1, // decode_lbrr = 1
                            cond_coding,
                        );
                        // Decode pulses (consume bitstream, output discarded)
                        let mut pulses = [0i16; MAX_FRAME_LENGTH];
                        silk_decode_pulses(
                            range_dec,
                            &mut pulses,
                            self.channel_state[0].indices.signal_type as i32,
                            self.channel_state[0].indices.quant_offset_type as i32,
                            self.channel_state[0].frame_length,
                        );
                    }
                }
            }
        }

        let mut n_samples_out: i32 = 0;
        let frame_index = self.channel_state[0].n_frames_decoded;
        let cond_coding = if frame_index == 0 {
            CODE_INDEPENDENTLY
        } else {
            CODE_CONDITIONALLY
        };

        let channel = &mut self.channel_state[0];
        let ret = silk_decode_frame(
            channel,
            range_dec,
            output,
            &mut n_samples_out,
            lost_flag,
            cond_coding,
        );

        channel.n_frames_decoded += 1;

        if ret < 0 { ret } else { n_samples_out }
    }

    /// Decode a SILK frame from raw bytes
    pub fn decode_bytes(&mut self, data: &[u8], output: &mut [i16], new_packet: bool) -> i32 {
        let mut range_dec = RangeCoder::new_decoder(data.to_vec());
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

    /// Reset the decoder state
    pub fn reset(&mut self) {
        silk_init_decoder(&mut self.channel_state[0]);
        silk_init_decoder(&mut self.channel_state[1]);
        self.prev_decode_only_middle = 0;
    }

    /// Get the frame length in samples
    pub fn frame_length(&self) -> i32 {
        self.channel_state[0].frame_length
    }

    /// Get the sample rate in Hz
    pub fn sample_rate(&self) -> i32 {
        self.channel_state[0].fs_khz * 1000
    }
}

#[cfg(test)]
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
        // SILK only supports 8, 12, or 16 kHz
        let ret = dec.init(16000, 1);
        assert_eq!(ret, 0);
        assert_eq!(dec.sample_rate(), 16000);
    }

    #[test]
    fn test_decoder_16khz() {
        let mut dec = SilkDecoder::new();
        let ret = dec.init(16000, 1);
        assert_eq!(ret, 0);
        assert_eq!(dec.frame_length(), 320); // 20ms at 16kHz
    }
}
