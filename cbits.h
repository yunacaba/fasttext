#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void *FastText_Handle_t;

    typedef struct
    {
        enum
        {
            SUCCESS = 0,
            ERROR = 1,
        } status;

        union {
            FastText_Handle_t handle;
            char *error;
        };
    } FastText_Result_t;

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

    FastText_Result_t FastText_NewHandle(const char *path);
    void FastText_DeleteHandle(const FastText_Handle_t handle);
    FastText_Predict_t FastText_Predict(const FastText_Handle_t handle, FastText_String_t query, int k,
                                        float threshold);
    FastText_Predict_t FastText_PredictOne(const FastText_Handle_t handle, FastText_String_t query, float threshold);

    FastText_FloatVector_t FastText_Wordvec(const FastText_Handle_t handle, FastText_String_t word);
    FastText_FloatVector_t FastText_Sentencevec(const FastText_Handle_t handle, FastText_String_t sentance);

    // char *FastText_Analogy(const FastText_Handle_t handle, FastText_String_t query);

    void FastText_FreeFloatVector(FastText_FloatVector_t vector);
    void FastText_FreePredict(FastText_Predict_t predict);

    FastText_PredictItem_t FastText_PredictItemAt(FastText_Predict_t predict, size_t idx);
#ifdef __cplusplus
}
#endif
