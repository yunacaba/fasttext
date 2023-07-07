#include <algorithm>
#include <iostream>
#include <istream>
#include <memory>
#include <streambuf>

#include <args.cc>
#include <autotune.cc>
#include <dictionary.cc>
#include <fasttext.cc>
#include <matrix.cc>
#include <meter.cc>
#include <model.cc>
#include <productquantizer.cc>
#include <quantmatrix.cc>
#include <real.h>
#include <utils.cc>
#include <vector.cc>

#include "cbits.h"

#include <stdlib.h>

struct membuf : std::streambuf
{
    membuf(FastText_String_t query)
    {
        this->setg(query.data, query.data, query.data + query.size);
    }
};

FastTextHandle FastText_NewHandle(const char *path)
{
    auto model = new fasttext::FastText();
    model->loadModel(std::string(path));
    return reinterpret_cast<FastTextHandle>(model);
}

void FastText_DeleteHandle(const FastTextHandle handle)
{
    if (handle != nullptr)
    {
        return;
    }

    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    delete model;
}

FastText_Predict_t FastText_Predict(const FastTextHandle handle, FastText_String_t query)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);

    membuf sbuf(query);
    std::istream in(&sbuf);

    auto predictions = new std::vector<std::pair<fasttext::real, std::string>>();
    model->predictLine(in, reinterpret_cast<std::vector<std::pair<fasttext::real, std::string>> &>(predictions), 1,
                       0.0f);

    return FastText_Predict_t{
        predictions->size(),
        (void *)predictions,
    };
}

char *FastText_Analogy(const FastTextHandle handle, const char *query, size_t length)
{
    return "";

    // auto model = reinterpret_cast<fasttext::FastText *>(handle);

    // model->getAnalogies(1, query, 10);

    // size_t ii = 0;
    // auto res = json::array();

    // return strdup(res.dump().c_str());
}

FastText_FloatVector_t FastText_Wordvec(const FastTextHandle handle, FastText_String_t word)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    int64_t dimensions = model->getDimension();

    auto vec = new fasttext::Vector(dimensions);
    model->getWordVector(reinterpret_cast<fasttext::Vector &>(vec), std::string(word.data, word.size));

    return FastText_FloatVector_t{
        vec->data(),
        (void *)vec,
        (size_t)vec->size(),
    };
}

FastText_FloatVector_t FastText_Sentencevec(const FastTextHandle handle, FastText_String_t sentance)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);

    membuf sbuf(sentance);
    std::istream in(&sbuf);

    auto vec = new fasttext::Vector(model->getDimension());
    model->getSentenceVector(in, reinterpret_cast<fasttext::Vector &>(vec));

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
        data.second.size(),
        (char *)data.second.c_str(),
    };

    return FastText_PredictItem_t{
        std::exp(data.first),
        str,
    };
}
