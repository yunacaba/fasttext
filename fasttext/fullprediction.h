#pragma once
#include "dictionary.h"
#include "utils.h"

namespace fasttext
{
class FullPrediction
{
  public:
    struct Item
    {
        std::string_view word;
        real score;
    };

  private:
    Predictions predictions_;
    std::shared_ptr<Dictionary> dict_;

  public:
    FullPrediction(Predictions &&predictions, const std::shared_ptr<Dictionary> &dict)
        : predictions_(std::move(predictions)), dict_(dict)
    {
    }

    Item at(size_t idx) const
    {
        const auto &prediction = predictions_.at(idx);

        const auto item = Item{
            .word = dict_->getLabelRef(prediction.second),
            .score = std::exp(prediction.first),
        };

        return item;
    }

    size_t size() const
    {
        return predictions_.size();
    }
};
} // namespace fasttext
