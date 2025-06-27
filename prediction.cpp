#include <algorithm>
#include <iomanip>
#include <iostream>
#include <istream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <streambuf>
#include <string_view>

#include <fasttext.h>

#include "predictions.h"

BEGIN_EXTERN_C()
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

END_EXTERN_C();
