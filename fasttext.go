package fasttext

// #cgo CXXFLAGS: -I${SRCDIR}/fasttext -I${SRCDIR} -std=c++17 -O3 -fPIC -pthread -march=native
// #cgo LDFLAGS: -lstdc++
// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include "cbits.h"
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

var (
	ErrPredictionFailed = errors.New("prediction failed")
	ErrNoPredictions    = errors.New("no predictions")
)

// A model object. Effectively a wrapper
// around the C fasttext handle
type (
	Model struct {
		p C.FastText_Handle_t
	}

	ModelOpenError string
)

func (e ModelOpenError) Error() string {
	return string(e)
}

// Open opens a model from a path and returns a model
// object
func Open(path string) (Model, error) {
	var pinner runtime.Pinner
	defer pinner.Unpin()

	cStrPath := cStr(path)
	pinner.Pin(cStrPath)
	result := C.FastText_NewHandle(C.FastText_String_t{
		data: cStrPath,
		size: C.size_t(len(path)),
	})

	if result.status != 0 {
		ch := *(**C.char)(unsafe.Pointer(&result.anon0[0]))
		defer C.free(unsafe.Pointer(ch))
		return Model{}, ModelOpenError(C.GoString(ch))
	}

	handle := *(*C.FastText_Handle_t)(unsafe.Pointer(&result.anon0[0]))

	return Model{
		p: handle,
	}, nil
}

// Closes a model handle
func (handle *Model) Close() error {
	C.FastText_DeleteHandle(handle.p)
	return nil
}

func (handle *Model) MultiLinePredict(lines []string, k int32, threshoad float32) ([]Predictions, error) {
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

// func (handle *Model) PredictOne(query string, threshoad float32) (Prediction, error) {
// 	r := C.FastText_Predict(
// 		handle.p,
// 		C.FastText_String_t{
// 			data: cStr(query),
// 			size: C.size_t(len(query)),
// 		},
// 		1,
// 		C.float(threshoad),
// 	)

// 	if r.data == nil {
// 		return Prediction{}, ErrPredictionFailed
// 	}

// 	defer C.FastText_FreePredict(r)

// 	if r.size == 0 {
// 		return Prediction{}, ErrNoPredictions
// 	}

// 	cPredic := C.FastText_PredictItemAt(r, C.size_t(0))

// 	return Prediction{
// 		Label:       C.GoStringN(cPredic.label.data, C.int(cPredic.label.size)),
// 		Probability: float32(cPredic.probability),
// 	}, nil
// }

// Perform model prediction
func (handle *Model) Predict(query string, k int32, threshoad float32) (Predictions, error) {
	var pinner runtime.Pinner
	defer pinner.Unpin()

	inputs := make([]C.FastText_PredictItem_t, k)
	inputsPtr := unsafe.SliceData(inputs)
	pinner.Pin(inputsPtr)

	count := C.FastText_Predict(
		handle.p,
		C.FastText_String_t{
			data: cStr(query),
			size: C.size_t(len(query)),
		},
		C.uint32_t(k),
		C.float(threshoad),
		inputsPtr,
	)

	if count == 0 {
		return nil, ErrNoPredictions
	}

	predictions := make(Predictions, k)

	for i := int32(0); i < k; i++ {
		str := C.GoStringN(inputs[i].lang.data, C.int(inputs[i].lang.size))

		predictions[i] = Prediction{
			Label:       str,
			Probability: float32(inputs[i].probability),
		}
	}

	return predictions, nil
}

// func (handle *Model) Analogy(word1, word2, word3 string, k int32) Analogs {
// 	// cWord1 := ((*C.char) unsafe.Pointer(unsafe.StringData(word1)))

// 	var pinner runtime.Pinner
// 	defer pinner.Unpin()

// 	pinner.Pin(word1)
// 	pinner.Pin(word2)
// 	pinner.Pin(word3)

// 	strWord1 := cStr(word1)
// 	pinner.Pin(strWord1)
// 	strWord2 := cStr(word2)
// 	pinner.Pin(strWord2)
// 	strWord3 := cStr(word3)
// 	pinner.Pin(strWord3)

// 	r := C.FastText_Analogy(
// 		handle.p,
// 		C.FastText_String_t{
// 			data: strWord1,
// 			size: C.size_t(len(word1)),
// 		},
// 		C.FastText_String_t{
// 			data: strWord2,
// 			size: C.size_t(len(word2)),
// 		},
// 		C.FastText_String_t{
// 			data: strWord3,
// 			size: C.size_t(len(word3)),
// 		},
// 		C.uint32_t(k),
// 	)

// 	defer C.FastText_FreePredict(r)

// 	analogs := make(Analogs, r.size)

// 	for i := uint64(0); i < uint64(r.size); i++ {
// 		cPredic := C.FastText_PredictItemAt(r, C.size_t(i))

// 		analogs[i] = Analog{
// 			Name:        C.GoStringN(cPredic.label.data, C.int(cPredic.label.size)),
// 			Probability: float32(cPredic.probability),
// 		}
// 	}

// 	return analogs
// }

// func (handle Model) Wordvec(word string) []float32 {
// 	var pinner runtime.Pinner
// 	defer pinner.Unpin()

// 	pinner.Pin(word)
// 	strData := cStr(word)
// 	pinner.Pin(strData)

// 	r := C.FastText_Wordvec(
// 		handle.p,
// 		C.FastText_String_t{
// 			data: strData,
// 			size: C.size_t(len(word)),
// 		},
// 	)
// 	defer C.FastText_FreeFloatVector(r)

// 	vectors := make([]float32, r.size)
// 	pinner.Pin(r.data)

// 	ptr := (*float32)(unsafe.Pointer(r.data))
// 	pinner.Pin(ptr)

// 	copy(vectors, unsafe.Slice(ptr, r.size))

// 	return vectors
// }

// Sentencevec requires sentence ends with </s>
// func (handle Model) Sentencevec(query string) []float32 {
// 	var pinner runtime.Pinner
// 	defer pinner.Unpin()
// 	pinner.Pin(query)
// 	strData := cStr(query)
// 	pinner.Pin(strData)
// 	r := C.FastText_Sentencevec(handle.p, C.FastText_String_t{
// 		data: strData,
// 		size: C.size_t(len(query)),
// 	})

// 	defer C.FastText_FreeFloatVector(r)

// 	vectors := make([]float32, r.size)
// 	pinner.Pin(r.data)
// 	ptr := (*float32)(unsafe.Pointer(r.data))
// 	pinner.Pin(ptr)
// 	copy(vectors, unsafe.Slice(ptr, r.size))

// 	return vectors
// }

func cStr(str string) *C.char {
	return (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
}
