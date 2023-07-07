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
