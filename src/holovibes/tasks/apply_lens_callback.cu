#include <cuComplex.h>

__device__ cuFloatComplex apply_lens_callback(void *data, size_t offset,
                                              void *callerInfo,
                                              void *sharedPtr) {
  size_t lens_idx = offset % (512 * 320);
  // offset %= 320 * 512;
  // if (offset >= 163840 - 1)
  //   printf("(%p) %llu\n", callerInfo, offset);

  cuFloatComplex *lens = (cuFloatComplex *)callerInfo;
  cuFloatComplex val = ((cuFloatComplex *)data)[offset];
  cuFloatComplex lens_val = lens[lens_idx];

  return cuCmulf(val, lens_val);
}