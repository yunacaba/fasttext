package fasttext_test

import (
	"testing"

	"github.com/nano-interactive/fasttext/v2"
	"github.com/stretchr/testify/require"
)

func TestOpen(t *testing.T) {
	t.Parallel()
	assert := require.New(t)

	t.Run("Success", func(t *testing.T) {
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

// func TestPredictOne(t *testing.T) {
// 	t.Parallel()
// 	assert := require.New(t)

// 	model, err := fasttext.Open("testdata/lid.176.ftz")
// 	assert.NoError(err)

// 	prediction, err := model.PredictOne("hello world from my dear C++", 0.0)

// 	assert.NoError(err)
// 	assert.NotEmpty(prediction)
// 	assert.Equal("en", prediction.Label)
// 	assert.Greater(prediction.Probability, float32(0.7))
// }

// func TestMultilinePredict(t *testing.T) {
// 	t.Parallel()
// 	assert := require.New(t)

// 	model, err := fasttext.Open("testdata/lid.176.ftz")

// 	assert.NoError(err)

// 	predictions, err := model.MultiLinePredict([]string{
// 		"Πες γεια στον μικρό μου φίλο",
// 		"Say 'ello to my little friend",
// 	}, 1, 0.5)

// 	assert.NoError(err)
// 	assert.NotEmpty(predictions)
// 	assert.Len(predictions, 2)

// 	assert.Len(predictions[0], 1)
// 	assert.Equal(predictions[0][0].Label, "el") // el => for greek
// 	assert.Len(predictions[1], 1)
// 	assert.Equal(predictions[1][0].Label, "en")
// }

func TestPredict(t *testing.T) {
	t.Parallel()
	assert := require.New(t)

	t.Run("WithOnePrediction", func(t *testing.T) {
		model, err := fasttext.Open("testdata/lid.176.ftz")

		assert.NoError(err)

		prediction, err := model.Predict(`
        This day is called the feast of Crispian.
        He that outlives this day, and comes safe home,
        Will stand a tip-toe when this day is named,
        And rouse him at the name of Crispian.
        He that shall live this day, and see old age,
        Will yearly on the vigil feast his neighbours,
        And say "To-morrow is Saint Crispian."
        Then will he strip his sleeve and show his scars,
        And say "These wounds I had on Crispin's day."
        Old men forget; yet all shall be forgot,
        But he'll remember, with advantages,
        What feats he did that day. Then shall our names,
        Familiar in his mouth as household words—
        Harry the King, Bedford and Exeter,
        Warwick and Talbot, Salisbury and Gloucester—
        Be in their flowing cups freshly remembered.
        This story shall the good man teach his son;
        And Crispin Crispian shall ne'er go by,
        From this day to the ending of the world,
        But we in it shall be remembered—
        We few, we happy few, we band of brothers;
        For he to-day that sheds his blood with me
        Shall be my brother; be he ne'er so vile,
        This day shall gentle his condition;
        And gentlemen in England now a-bed
        Shall think themselves accurs'd they were not here,
        And hold their manhoods cheap whiles any speaks
        That fought with us upon Saint Crispin's day.
    `, 1, 0.7)

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
