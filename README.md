# vAccelRT

[![Build project](https://github.com/cloudkernels/vaccelrt/actions/workflows/test_vaccelrt.yml/badge.svg)](https://github.com/cloudkernels/vaccelrt/actions/workflows/test_vaccelrt.yml)
[![Build Deb Package](https://github.com/cloudkernels/vaccelrt/actions/workflows/deb.yml/badge.svg)](https://github.com/cloudkernels/vaccelrt/actions/workflows/deb.yml)

vAccelRT is a runtime library for hardware acceleration. vAccelRT provides an
API with a set of functions that the library is able to offload to hardware
acceleration devices. The design of the runtime library is modular, it consists
of a front-end library which exposes the API to the user application and a set
of backend plugins that are responsible to offload computations to the
accelerator.

This design decouples the user application from the actual accelerator specific
code. The advantage of this choice is that the application can make use of
different hardware accelerators without extra development cost or re-compiling.

This repo includes the core runtime library, and a backend plugin for the
`EXEC` operation. For debugging and demonstration purposes we include a `NOOP`
plugin which just prints out debug parameters (input and output) for each API
call.

There is a [splash page](https://vaccel.org) for vAccel, along with more
[elaborate documentation](https://docs.vaccel.org). 

For step-by-step tutorials, you can have a look at our
[lab](https://github.com/nubificus/vaccel-tutorials) repo.


## Image Classification running on TPU

This fork of vAccelRT contains an extra image-classification operation, along with a new TPU-plugin. Inside "examples" directory, you can find tpu_img_class.c, that runs an inference. The plugin uses classify.h, a [library](https://github.com/ilagomatis/libcoral) that let us run image classification using Edge TPU accelerator.

## Operation

## TPU-Plugin

## License

[Apache License 2.0](LICENSE)
