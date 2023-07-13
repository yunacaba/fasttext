package fasttext

// #cgo CXXFLAGS: -I${SRCDIR}/fastText-src -I${SRCDIR} -std=c++17 -O3 -fPIC -pthread -march=native
// #cgo LDFLAGS: -lstdc++
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits.h"
import "C"

import (
	"errors"
	"unsafe"
)

var (
	ErrPredictionFailed = errors.New("prediction failed")
	ErrNoPredictions    = errors.New("no predictions")
)

// A model object. Effectively a wrapper
// around the C fasttext handle
type Model struct {
	p C.FastText_Handle_t
}

type ModelOpenError struct {
	val string
}

func (e *ModelOpenError) Error() string {
	return e.val
}

// Opens a model from a path and returns a model
// object
func Open(path string) (Model, error) {
	result := C.FastText_NewHandle(C.FastText_String_t{
		data: cStr(path),
		size: C.size_t(len(path)),
	})

	if result.status != 0 {
		ch := *(**C.char)(unsafe.Pointer(&result.anon0[0]))
		defer C.free(unsafe.Pointer(ch))
		return Model{}, &ModelOpenError{
			val: C.GoString(ch),
		}
	}

	handle := *(*C.FastText_Handle_t)(unsafe.Pointer(&result.anon0[0]))

	return Model{
		p: handle,
	}, nil
}

// Closes a model handle
func (handle *Model) Close() error {
	if handle == nil {
		return nil
	}
	C.FastText_DeleteHandle(handle.p)
	return nil
}

func (handle Model) MultiLinePredict(lines []string, k int32, threshoad float32) ([]Predictions, error) {
	predics := make([]Predictions, 0, len(lines))

	for _, line := range lines {
		predictions, err := handle.Predict(line, k, threshoad)
		if err != nil && errors.Is(err, ErrPredictionFailed) {
			return nil, err
		}

		predics = append(predics, predictions)
	}

	if len(predics) == 0 {
		return nil, ErrNoPredictions
	}

	return predics, nil
}

func (handle Model) PredictOne(query string, threshoad float32) (Prediction, error) {
	r := C.FastText_Predict(
		handle.p,
		C.FastText_String_t{
			data: cStr(query),
			size: C.size_t(len(query)),
		},
		1,
		C.float(threshoad),
	)

	if r.data == nil {
		return Prediction{}, ErrPredictionFailed
	}

	defer C.FastText_FreePredict(r)

	if r.size == 0 {
		return Prediction{}, ErrNoPredictions
	}

	cPredic := C.FastText_PredictItemAt(r, C.size_t(0))

	return Prediction{
		Label:       C.GoStringN(cPredic.label.data, C.int(cPredic.label.size)),
		Probability: float32(cPredic.probability),
	}, nil
}

// Perform model prediction
func (handle Model) Predict(query string, k int32, threshoad float32) (Predictions, error) {
	r := C.FastText_Predict(
		handle.p,
		C.FastText_String_t{
			data: cStr(query),
			size: C.size_t(len(query)),
		},
		C.int(k),
		C.float(threshoad),
	)

	if r.data == nil {
		return nil, ErrPredictionFailed
	}

	defer C.FastText_FreePredict(r)

	if r.size == 0 {
		return nil, ErrNoPredictions
	}

	predictions := make(Predictions, r.size)

	for i := 0; i < int(r.size); i++ {
		cPredic := C.FastText_PredictItemAt(r, C.size_t(i))

		predictions[i] = Prediction{
			Label:       C.GoStringN(cPredic.label.data, C.int(cPredic.label.size)),
			Probability: float32(cPredic.probability),
		}
	}

	return predictions, nil
}

func (handle Model) Analogy(word1, word2, word3 string, k int32) Analogs {
	// cWord1 := ((*C.char) unsafe.Pointer(unsafe.StringData(word1)))

	r := C.FastText_Analogy(
		handle.p,
		C.FastText_String_t{
			data: cStr(word1),
			size: C.size_t(len(word1)),
		},
		C.FastText_String_t{
			data: cStr(word2),
			size: C.size_t(len(word2)),
		},
		C.FastText_String_t{
			data: cStr(word3),
			size: C.size_t(len(word3)),
		},
		C.int32_t(k),
	)

	defer C.FastText_FreePredict(r)

	analogs := make(Analogs, r.size)

	for i := uint64(0); i < uint64(r.size); i++ {
		cPredic := C.FastText_PredictItemAt(r, C.size_t(i))

		analogs[i] = Analog{
			Name:        C.GoStringN(cPredic.label.data, C.int(cPredic.label.size)),
			Probability: float32(cPredic.probability),
		}
	}

	return analogs
}

func (handle Model) Wordvec(word string) []float32 {
	r := C.FastText_Wordvec(
		handle.p,
		C.FastText_String_t{
			data: cStr(word),
			size: C.size_t(len(word)),
		},
	)
	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	copy(vectors, unsafe.Slice((*float32)(unsafe.Pointer(r.data)), r.size))

	return vectors
}

// Requires sentence ends with </s>
func (handle Model) Sentencevec(query string) []float32 {
	r := C.FastText_Sentencevec(handle.p, C.FastText_String_t{
		data: cStr(query),
		size: C.size_t(len(query)),
	})

	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	copy(vectors, unsafe.Slice((*float32)(unsafe.Pointer(r.data)), r.size))

	return vectors
}

//go:inline
func cStr(str string) *C.char {
	return ((*C.char)(unsafe.Pointer(unsafe.StringData(str))))
}
