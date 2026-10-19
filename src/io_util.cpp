#include "io_util.hpp"

#include "ggml.h"
#include "ggml-backend.h"

namespace io {

bool seek64(FILE * f, int64_t off, int origin) {
#ifdef _WIN32
    return _fseeki64(f, off, origin) == 0;
#else
    return fseeko(f, (off_t) off, origin) == 0;
#endif
}

int64_t tell64(FILE * f) {
#ifdef _WIN32
    return _ftelli64(f);
#else
    return (int64_t) ftello(f);
#endif
}

const int32_t * host_ids(ggml_tensor * t, std::vector<uint8_t> & scratch) {
    if (ggml_backend_buffer_is_host(t->buffer)) {
        return (const int32_t *) t->data;
    }
    const size_t nb = ggml_nbytes(t);
    scratch.resize(nb);
    ggml_backend_tensor_get(t, scratch.data(), 0, nb);
    return (const int32_t *) scratch.data();
}

}  // namespace io
