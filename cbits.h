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

    typedef struct
    {
        size_t size;
        char *data;
    } FastText_String_t;

    typedef struct
    {
        float probability;
        FastText_String_t label;
    } FastText_PredictItem_t;

    typedef struct
    {
        size_t size;
        void *data;
    } FastText_Predict_t;

    FastTextHandle FastText_NewHandle(const char *path);
    void FastText_DeleteHandle(const FastTextHandle handle);
    FastText_Predict_t FastText_Predict(const FastTextHandle handle, FastText_String_t query);
    FastText_FloatVector_t FastText_Wordvec(const FastTextHandle handle, FastText_String_t word);
    FastText_FloatVector_t FastText_Sentencevec(const FastTextHandle handle, FastText_String_t sentance);

    char *FastText_Analogy(const FastTextHandle handle, FastText_String_t query);

    void FastText_FreeFloatVector(FastText_FloatVector_t vector);
    void FastText_FreePredict(FastText_Predict_t predict);

    FastText_PredictItem_t FastText_PredictItemAt(FastText_Predict_t predict, size_t idx);
#ifdef __cplusplus
}
#endif
