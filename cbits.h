#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void *FastTextHandle;

    typedef struct
    {
        float *data;
        void *handle;
        size_t size;
    } FastText_FloatVector_t;

    FastTextHandle FastText_NewHandle(const char *path);
    void FastText_DeleteHandle(const FastTextHandle handle);
    char *FastText_Predict(const FastTextHandle handle, const char *query, size_t length);
    char *FastText_Analogy(const FastTextHandle handle, const char *query, size_t length);
    FastText_FloatVector_t FastText_Wordvec(const FastTextHandle handle, const char *word, size_t length);
    FastText_FloatVector_t FastText_Sentencevec(const FastTextHandle handle, const char *sentance, size_t length);

    void FastText_FreeFloatVector(FastText_FloatVector_t vector);

#ifdef __cplusplus
}
#endif
