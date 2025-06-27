#include <cstring>
#include <string>

#include <fasttext.h>

#include "predictions.h"

BEGIN_EXTERN_C()

FastText_Result_t FastText_NewHandle(FastText_String_t path)
{
    auto model = new fasttext::FastText();

    try
    {
        model->loadModel(std::string(path.data, path.size));
        return FastText_Result_t{
            FastText_Result_t::SUCCESS,
            (FastText_Handle_t)model,
        };
    }
    catch (std::exception &e)
    {
        return FastText_Result_t{
            FastText_Result_t::ERROR,
            strdup(e.what()),
        };
    }
}

void FastText_DeleteHandle(const FastText_Handle_t handle)
{
    if (handle != nullptr)
    {
        return;
    }

    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    delete model;
}
END_EXTERN_C()
