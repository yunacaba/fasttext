#include <iostream>
#include <istream>
#include <memory>
#include <streambuf>

#include <json.hpp>

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

using json = nlohmann::json;

#include <stdlib.h>

struct membuf : std::streambuf
{
    membuf(const char *begin, const char *end)
    {
        this->setg((char *)begin, (char *)begin, (char *)end);
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

char *FastText_Predict(const FastTextHandle handle, const char *query, size_t length)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);

    membuf sbuf(query, query + length);
    std::istream in(&sbuf);

    std::vector<std::pair<fasttext::real, std::string>> predictions;
    model->predictLine(in, predictions, 1, 0.0f);

    size_t ii = 0;
    auto res = json::array();
    for (const auto it : predictions)
    {
        float p = std::exp(it.first);
        res.push_back({
            {"index", ii++},
            {"probability", p},
            {"label", it.second},
        });
    }

    return strdup(res.dump().c_str());
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

FastText_FloatVector_t FastText_Wordvec(const FastTextHandle handle, const char *word, size_t length)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    int64_t dimensions = model->getDimension();

    auto vec = new fasttext::Vector(dimensions);
    model->getWordVector(reinterpret_cast<fasttext::Vector &>(vec), word);

    return FastText_FloatVector_t{
        vec->data(),
        (void *)vec,
        (size_t)vec->size(),
    };
}

FastText_FloatVector_t FastText_Sentencevec(const FastTextHandle handle, const char *sentance, size_t length)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);

    membuf sbuf(sentance, sentance + length);
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
