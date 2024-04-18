package fasttext

import "C"

import "unsafe"

func cStr(str string) *C.char {
	return (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
}
