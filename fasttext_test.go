package fasttext_test

import (
	"testing"

	"github.com/nano-interactive/go-fasttext"
	"github.com/stretchr/testify/require"
)

func TestOpen(t *testing.T) {
	t.Parallel()
	assert := require.New(t)

	t.Run("Sucess", func(t *testing.T) {
		model, err := fasttext.Open("testdata/lid.176.ftz")

		assert.NoError(err)
		assert.NotEmpty(model)
		assert.NoError(model.Close())
	})

  t.Run("FailedToOpen", func(t *testing.T) {
    model, err := fasttext.Open("testdata/lid-not-found.176.ftz")

		assert.EqualError(err, "testdata/lid-not-found.176.ftz cannot be opened for loading!")
		assert.Empty(model)
  })
}

func TestPredictOne(t *testing.T) {
	t.Parallel()
	assert := require.New(t)

	model, err := fasttext.Open("testdata/lid.176.ftz")
	assert.NoError(err)

	prediction, err := model.PredictOne("hello world from my dear C++", 0.0)

	assert.NoError(err)
	assert.NotEmpty(prediction)
	assert.Equal("en", prediction.Label)
	assert.Greater(prediction.Probability, float32(0.7))
}

func TestMultilinePredict(t *testing.T) {
	t.Parallel()
	assert := require.New(t)

	model, err := fasttext.Open("testdata/lid.176.ftz")

	assert.NoError(err)

	predictions, err := model.MultiLinePredict([]string{
		"Πες γεια στον μικρό μου φίλο",
		"Say 'ello to my little friend",
	}, 1, 0.5)

	assert.NoError(err)
	assert.NotEmpty(predictions)
	assert.Len(predictions, 2)

	assert.Len(predictions[0], 1)
	assert.Equal(predictions[0][0].Label, "el") // el => for greek
	assert.Len(predictions[1], 1)
	assert.Equal(predictions[1][0].Label, "en")
}

func TestPredict(t *testing.T) {
	t.Parallel()
	assert := require.New(t)

	t.Run("WithOnePrediction", func(t *testing.T) {
		model, err := fasttext.Open("testdata/lid.176.ftz")

		assert.NoError(err)

		prediction, err := model.Predict("hello world from my dear C++", 1, 0.7)

		assert.NoError(err)
		assert.NotEmpty(prediction)
		assert.Len(prediction, 1)
		assert.Equal("en", prediction[0].Label)
		assert.Greater(prediction[0].Probability, float32(0.7))
	})

	t.Run("WithMultiple", func(t *testing.T) {
		model, err := fasttext.Open("testdata/lid.176.ftz")

		assert.NoError(err)

		prediction, err := model.Predict("hello", 3, 0.0)

		assert.NoError(err)
		assert.NotEmpty(prediction)
		assert.Len(prediction, 3)
		assert.Equal("en", prediction[0].Label)
		assert.Equal("fr", prediction[1].Label)
		assert.Equal("ru", prediction[2].Label)
	})

	t.Run("Gibberish", func(t *testing.T) {
		model, err := fasttext.Open("testdata/lid.176.ftz")

		assert.NoError(err)

		prediction, err := model.Predict("asdasd asdasdasd asd ", 3, 0.0)

		assert.ErrorIs(err, fasttext.ErrNoPredictions)
		assert.Nil(prediction)
	})
}
