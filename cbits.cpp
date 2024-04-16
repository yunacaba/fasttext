#include <algorithm>
#include <iomanip>
#include <iostream>
#include <istream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <streambuf>
#include <string_view>

#include <stdlib.h>

#include <args.cc>
#include <autotune.cc>
#include <densematrix.cc>
#include <dictionary.cc>
#include <loss.cc>
#include <matrix.cc>
#include <meter.cc>
#include <model.cc>
#include <productquantizer.cc>
#include <quantmatrix.cc>
#include <utils.cc>
#include <vector.cc>

#include "cbits.h"

#define FREE_STRING(str)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (str.data != nullptr)                                                                                       \
            free(str.data);                                                                                            \
        str.data = nullptr;                                                                                            \
        str.size = 0;                                                                                                  \
    } while (0)

#define LABEL_PREFIX ("__label__")
#define LABEL_PREFIX_SIZE (sizeof(LABEL_PREFIX) - 1)

using Predictions = std::vector<std::pair<fasttext::real, std::string>>;

struct membuf : std::streambuf
{
    membuf(FastText_String_t query)
    {
        this->setg(query.data, query.data, query.data + query.size);
    }
};

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

size_t FastText_Predict(const FastText_Handle_t handle, FastText_String_t query, uint32_t k, float threshold,
                        FastText_PredictItem_t *const value)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    auto predictions = model->predictFull(k, std::string_view(query.data, query.size), threshold);
    const auto count = k > predictions.size() ? predictions.size() : k;

    for (size_t i = 0; i < count; i++)
    {
        const auto &prediction = predictions.at(i);

        std::string_view data = prediction.word.substr(LABEL_PREFIX_SIZE);
        size_t size = data.size();

        if (size > 8)
        {
            size = 8;
        }

        value[i].probability = prediction.score;
        value[i].lang = FastText_String_t{
            .size = size,
            .data = (char *)data.data(),
        };
    }

    return count;
}

// FastText_Predict_t FastText_Analogy(const FastText_Handle_t handle, FastText_String_t word1, FastText_String_t word2,
//                                     FastText_String_t word3, uint32_t k)
// {
//     const auto model = reinterpret_cast<fasttext::FastText *>(handle);
//     Predictions predictions =
//         model->getAnalogies(k, std::string(word1.data, word1.size), std::string(word2.data, word2.size),
//                             std::string(word3.data, word3.size));

//     auto vec = new Predictions(std::move(predictions));

//     return FastText_Predict_t{
//         vec->size(),
//         (void *)vec,
//     };
// }

FastText_FloatVector_t FastText_Wordvec(const FastText_Handle_t handle, FastText_String_t word)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    int64_t dimensions = model->getDimension();

    auto vec = new fasttext::Vector(dimensions);
    model->getWordVector(*vec, std::string(word.data, word.size));

    return FastText_FloatVector_t{
        vec->data(),
        (void *)vec,
        (size_t)vec->size(),
    };
}

FastText_FloatVector_t FastText_Sentencevec(const FastText_Handle_t handle, FastText_String_t sentence)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);

    membuf sbuf(sentence);
    std::istream in(&sbuf);

    auto vec = new fasttext::Vector(model->getDimension());
    model->getSentenceVector(in, *vec);
    FREE_STRING(sentence);

    return FastText_FloatVector_t{
        vec->data(),
        (void *)vec,
        (size_t)vec->size(),
    };
}

void FastText_FreeFloatVector(FastText_FloatVector_t vector)
{
    auto vec = reinterpret_cast<fasttext::Vector *>(vector.handle);
    delete vec;
}

void FastText_FreePredict(FastText_Predict_t predict)
{
    auto vec = reinterpret_cast<Predictions *>(predict.data);
    delete vec;
}

END_EXTERN_C()
