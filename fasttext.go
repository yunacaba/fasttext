package fasttext

// #cgo CXXFLAGS: -I${SRCDIR}/fastText/src -I${SRCDIR} -I${SRCDIR}/include -std=c++17 -O3 -fPIC -pedantic -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter
// #cgo LDFLAGS: -lstdc++
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits.h"
import "C"

import (
	"unsafe"
)

// A model object. Effectively a wrapper
// around the C fasttext handle
type Model struct {
	p C.FastTextHandle
}

// Opens a model from a path and returns a model
// object
func Open(path string) Model {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	return Model{
		p: C.FastText_NewHandle(cpath),
	}
}

// Closes a model handle
func (handle *Model) Close() error {
	if handle == nil {
		return nil
	}
	C.FastText_DeleteHandle(handle.p)
	return nil
}

// // Perform model prediction
func (handle Model) Predict(query string) Predictions {
	cquery := C.CString(query)
	defer C.free(unsafe.Pointer(cquery))

	// Call the Predict function defined in cbits.cpp
	// passing in the model handle and the query string
	r := C.FastText_Predict(handle.p, C.FastText_String_t{
		data: cquery,
		size: C.size_t(len(query)),
	})
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

// func (handle Model) Analogy(query string) (Analogs, error) {
// 	cquery := C.CString(query)
// 	defer C.free(unsafe.Pointer(cquery))

// 	r := C.Analogy(handle.handle, cquery, C.size_t(len(query)))
// 	defer C.free(unsafe.Pointer(r))
// 	js := C.GoString(r)

// 	analogies := []Analog{}
// 	err := json.Unmarshal([]byte(js), &analogies)
// 	if err != nil {
// 		return nil, err
// 	}

// 	return analogies, nil
// }

func (handle Model) Wordvec(word string) []float32 {
	cquery := C.CString(word)
	defer C.free(unsafe.Pointer(cquery))

	r := C.FastText_Wordvec(handle.p, C.FastText_String_t{
		data: cquery,
		size: C.size_t(len(word)),
	})
	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	copy(vectors, unsafe.Slice((*float32)(unsafe.Pointer(r.data)), r.size))

	return vectors
}

// Requires sentence ends with </s>
func (handle Model) Sentencevec(query string) []float32 {
	cquery := C.CString(query)
	defer C.free(unsafe.Pointer(cquery))

	r := C.FastText_Sentencevec(handle.p, C.FastText_String_t{
		data: cquery,
		size: C.size_t(len(query)),
	})

	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	copy(vectors, unsafe.Slice((*float32)(unsafe.Pointer(r.data)), r.size))

	return vectors
}
