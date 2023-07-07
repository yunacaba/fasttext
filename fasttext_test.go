package fasttext_test

import (
	"testing"

	"github.com/nano-interactive/go-fasttext"
	"github.com/stretchr/testify/require"
)

func TestOpen(t *testing.T) {
	t.Parallel()
	assert := require.New(t)

	_, err := fasttext.Open("testdata/lid.176.ftz")

	assert.NoError(err)
}


func TestPredictOne(t *testing.T) {

	assert := require.New(t)

	model, err := fasttext.Open("testdata/lid.176.ftz")

	assert.NoError(err)

  prediction := model.PredictOne("hello world from my dear C++", 0.7)

  assert.Equal("en", prediction.Label)
  assert.Greater(prediction.Probability, float32(0.7))
}
