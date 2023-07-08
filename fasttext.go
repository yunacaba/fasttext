package fasttext

// #cgo CXXFLAGS: -I${SRCDIR}/fastText-src -I${SRCDIR} -std=c++17 -O3 -fPIC -pthread -march=native
// #cgo LDFLAGS: -lstdc++
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits.h"
import "C"

import (
	"strings"
	"unsafe"
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
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	result := C.FastText_NewHandle(cpath)

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

func (handle Model) MultiLinePredict(query string, k int32, threshoad float32) []Predictions {
	lines := strings.Split(query, "\n")

	predics := make([]Predictions, 0, len(lines))

	for _, line := range lines {
		predictions := handle.Predict(line, k, threshoad)
		predics = append(predics, predictions)
	}

	return predics
}

func (handle Model) PredictOne(query string, threshoad float32) Prediction {
	r := C.FastText_PredictOne(
		handle.p,
		C.FastText_String_t{
			data: C.CString(query),
		},
		C.float(threshoad),
	)
	defer C.FastText_FreePredict(r)

	cPredic := C.FastText_PredictItemAt(r, C.size_t(0))

	return Prediction{
		Label:       C.GoStringN(cPredic.label.data, C.int(cPredic.label.size)),
		Probability: float32(cPredic.probability),
	}
}

// Perform model prediction
func (handle Model) Predict(query string, k int32, threshoad float32) Predictions {
	r := C.FastText_Predict(
		handle.p,
		C.FastText_String_t{
			data: C.CString(query),
			size: C.size_t(len(query)),
		},
		C.int(k),
		C.float(threshoad),
	)
	defer C.FastText_FreePredict(r)

	predictions := make(Predictions, r.size)

	for i := 0; i < int(r.size); i++ {
		cPredic := C.FastText_PredictItemAt(r, C.size_t(i))

		predictions[i] = Prediction{
			Label:       C.GoStringN(cPredic.label.data, C.int(cPredic.label.size)),
			Probability: float32(cPredic.probability),
		}
	}

	return predictions
}

func (handle Model) Analogy(word1, word2, word3 string, k int32) Analogs {
	r := C.FastText_Analogy(
		handle.p,
		C.FastText_String_t{
			data: C.CString(word1),
		},
		C.FastText_String_t{
			data: C.CString(word2),
		},
		C.FastText_String_t{
			data: C.CString(word3),
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
			data: C.CString(word),
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
		data: C.CString(query),
		size: C.size_t(len(query)),
	})

	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	copy(vectors, unsafe.Slice((*float32)(unsafe.Pointer(r.data)), r.size))

	return vectors
}
