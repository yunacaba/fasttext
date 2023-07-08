#include <algorithm>
#include <iomanip>
#include <iostream>
#include <istream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <streambuf>

#include <stdlib.h>

#include <loss.cc>
#include <args.cc>
#include <autotune.cc>
#include <densematrix.cc>
#include <dictionary.cc>
#include <matrix.cc>
#include <meter.cc>
#include <model.cc>
#include <productquantizer.cc>
#include <quantmatrix.cc>
#include <utils.cc>
#include <vector.cc>

#include "cbits.h"

struct membuf : std::streambuf
{
    membuf(FastText_String_t query)
    {
        this->setg(query.data, query.data, query.data + query.size);
    }
};

FastText_Result_t FastText_NewHandle(const char *path)
{
    auto model = new fasttext::FastText();

    try
    {
        model->loadModel(std::string(path));
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

FastText_Predict_t FastText_PredictOne(const FastText_Handle_t handle, FastText_String_t query, float threshold)
{
    return FastText_Predict(handle, query, 1, threshold);
}

FastText_Predict_t FastText_Predict(const FastText_Handle_t handle, FastText_String_t query, int k, float threshold)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);

    membuf sbuf(query);
    std::istream in(&sbuf);

    auto predictions = new std::vector<std::pair<fasttext::real, std::string>>();
    model->predictLine(in, *predictions, k, threshold);

    free(query.data);
    query.data = nullptr;
    query.size = 0;

    return FastText_Predict_t{
        predictions->size(),
        (void *)predictions,
    };
}

// char *FastText_Analogy(const FastText_Handle_t handle, const char *query, size_t length)
// {
//     return "";

//     // auto model = reinterpret_cast<fasttext::FastText *>(handle);

//     // model->getAnalogies(1, query, 10);

//     // size_t ii = 0;
//     // auto res = json::array();

//     // return strdup(res.dump().c_str());
// }

FastText_FloatVector_t FastText_Wordvec(const FastText_Handle_t handle, FastText_String_t word)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    int64_t dimensions = model->getDimension();

    auto vec = new fasttext::Vector(dimensions);
    model->getWordVector(*vec, std::string(word.data, word.size));

    free(word.data);
    word.data = nullptr;
    word.size = 0;

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
    free(sentence.data);
    sentence.data = nullptr;
    sentence.size = 0;

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
    auto vec = reinterpret_cast<std::vector<std::pair<fasttext::real, std::string>> *>(predict.data);
    delete vec;
}

FastText_PredictItem_t FastText_PredictItemAt(FastText_Predict_t predict, size_t idx)
{
    const auto vec = reinterpret_cast<std::vector<std::pair<fasttext::real, std::string>> *>(predict.data);
    const auto &data = vec->at(idx);

    auto str = FastText_String_t{
        data.second.size() - sizeof("__label__") + 1,
        (char *)(data.second.c_str() + sizeof("__label__") - 1),
    };

    return FastText_PredictItem_t{
        data.first,
        str,
    };
}
