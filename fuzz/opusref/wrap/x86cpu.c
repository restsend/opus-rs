/* libopus's x86 CPU detection with a level the tests can force:
   opus_select_arch() returns opusref_forced_arch when it is 0 or above.
   Every codec state takes its arch from opus_select_arch() at init. */
#define opus_select_arch opus_select_arch_detected
#include "celt/x86/x86cpu.c"
#undef opus_select_arch

int opusref_forced_arch = -1;

int opus_select_arch(void) {
    return opusref_forced_arch >= 0 ? opusref_forced_arch : opus_select_arch_detected();
}
