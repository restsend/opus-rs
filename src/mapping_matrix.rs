//! libopus's mapping matrices (`src/mapping_matrix.c`): the mixing of the
//! projection encoder's input channels into its coded channels and the
//! demixing of the decoded ones into the output, in Q15, stored column by
//! column (`data[rows * col + row]`).

mod tables;

pub(crate) use tables::*;

/// libopus `MappingMatrix`: `rows` output and `cols` input channels, the
/// gain in Q8 dB, and the cells column by column.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MappingMatrix<'a> {
    pub rows: usize,
    pub cols: usize,
    pub gain: i32,
    pub data: &'a [i16],
}

impl MappingMatrix<'_> {
    fn cell(&self, row: usize, col: usize) -> i16 {
        self.data[self.rows * col + row]
    }

    /// `mapping_matrix_multiply_channel_in_float`: output channel
    /// `output_row` of interleaved float `input` (`input_rows` channels),
    /// written to every `output_rows`-th sample of `output`.
    pub fn in_float(&self, input: &[f32], input_rows: usize, output: &mut [f32], output_row: usize, output_rows: usize, frame_size: usize) {
        for i in 0..frame_size {
            let mut tmp = 0.0f32;
            for col in 0..input_rows {
                tmp += self.cell(output_row, col) as f32 * input[input_rows * i + col];
            }
            output[output_rows * i] = (1.0 / 32768.0) * tmp;
        }
    }

    /// `mapping_matrix_multiply_channel_in_short` of the float build.
    pub fn in_short(&self, input: &[i16], input_rows: usize, output: &mut [f32], output_row: usize, output_rows: usize, frame_size: usize) {
        for i in 0..frame_size {
            let mut tmp = 0.0f32;
            for col in 0..input_rows {
                tmp += (self.cell(output_row, col) as i32 * input[input_rows * i + col] as i32) as f32;
            }
            output[output_rows * i] = (1.0 / (32768.0 * 32768.0)) * tmp;
        }
    }

    /// `mapping_matrix_multiply_channel_in_int24` of the float build.
    pub fn in_int24(&self, input: &[i32], input_rows: usize, output: &mut [f32], output_row: usize, output_rows: usize, frame_size: usize) {
        for i in 0..frame_size {
            // opus_val64 is float in the float build.
            let mut tmp = 0.0f32;
            for col in 0..input_rows {
                tmp += self.cell(output_row, col) as f32 * input[input_rows * i + col] as f32;
            }
            // INT24TORES((1/32768.f)*tmp).
            output[output_rows * i] = (1.0 / 32768.0 / 256.0) * ((1.0 / 32768.0) * tmp);
        }
    }

    /// `mapping_matrix_multiply_channel_out_float`: adds decoded channel
    /// `input_row` of `input` (every `input_rows`-th sample) to every
    /// output channel of interleaved `output` (`output_rows` channels).
    pub fn out_float(&self, input: &[f32], input_row: usize, input_rows: usize, output: &mut [f32], output_rows: usize, frame_size: usize) {
        for i in 0..frame_size {
            let x = input[input_rows * i];
            for row in 0..output_rows {
                output[output_rows * i + row] += (1.0 / 32768.0) * self.cell(row, input_row) as f32 * x;
            }
        }
    }

    /// `mapping_matrix_multiply_channel_out_short` (the 16-bit output
    /// wraps as libopus's does).
    pub fn out_short(&self, input: &[f32], input_row: usize, input_rows: usize, output: &mut [i16], output_rows: usize, frame_size: usize) {
        for i in 0..frame_size {
            let x = crate::float2int16(input[input_rows * i]) as i32;
            for row in 0..output_rows {
                let tmp = self.cell(row, input_row) as i32 * x;
                let o = &mut output[output_rows * i + row];
                *o = o.wrapping_add(((tmp + 16384) >> 15) as i16);
            }
        }
    }

    /// `mapping_matrix_multiply_channel_out_int24`.
    pub fn out_int24(&self, input: &[f32], input_row: usize, input_rows: usize, output: &mut [i32], output_rows: usize, frame_size: usize) {
        use crate::decode_int::IntSample;
        for i in 0..frame_size {
            let x = i32::from_float(input[input_rows * i]) as i64;
            for row in 0..output_rows {
                let tmp = self.cell(row, input_row) as i64 * x;
                let o = &mut output[output_rows * i + row];
                *o = o.wrapping_add(((tmp + 16384) >> 15) as i32);
            }
        }
    }
}
