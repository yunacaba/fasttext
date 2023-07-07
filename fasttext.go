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
	// fmt.Println("something")
	// create a C string from the Go string
	cpath := C.CString(path)
	// you have to delete the converted string
	// See https://github.com/golang/go/wiki/cgo
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

// // Performs model prediction
// func (handle *Model) Predict(query string) (Predictions, error) {
// 	cquery := C.CString(query)
// 	defer C.free(unsafe.Pointer(cquery))

// 	// Call the Predict function defined in cbits.cpp
// 	// passing in the model handle and the query string
// 	r := C.Predict(handle.handle, cquery, C.size_t(len(query)))
// 	// the C code returns a c string which we need to
// 	// convert to a go string
// 	defer C.free(unsafe.Pointer(r))
// 	js := C.GoString(r)

// 	// unmarshal the json results into the predictions
// 	// object. See https://blog.golang.org/json-and-go
// 	predictions := []Prediction{}
// 	err := json.Unmarshal([]byte(js), &predictions)
// 	if err != nil {
// 		return nil, err
// 	}

// 	return predictions, nil
// }

// func (handle *Model) Analogy(query string) (Analogs, error) {
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

func (handle *Model) Wordvec(word string) []float32 {
	cquery := C.CString(word)
	defer C.free(unsafe.Pointer(cquery))

	r := C.FastText_Wordvec(handle.p, cquery, C.size_t(len(word)))
	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	copy(vectors, unsafe.Slice((*float32)(unsafe.Pointer(r.data)), r.size))

	return vectors
}

// Requires sentence ends with </s>
func (handle *Model) Sentencevec(query string) []float32 {
  cquery := C.CString(query)
  defer C.free(unsafe.Pointer(cquery))

  r := C.FastText_Sentencevec(handle.p, cquery, C.size_t(len(query)))
	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	copy(vectors, unsafe.Slice((*float32)(unsafe.Pointer(r.data)), r.size))

	return vectors
}
