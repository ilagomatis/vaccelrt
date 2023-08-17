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

```c
#ifndef __VACCEL_IMAGE_CLASSIFICATION_H__
#define __VACCEL_IMAGE_CLASSIFICATION_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vaccel_session;

int vaccel_custom_image_classification(struct vaccel_session *sess,
				       char* model_path,
                		       char* image_path,
                		       char* labels_path,
                		       float input_mean,
                		       float input_std,
	       			       char* output);

#ifdef __cplusplus
}
#endif

#endif /* __VACCEL_IMAGE_CLASSIFICATION_H__ */
```

```c
#ifndef __IMAGE_CLASSIFICATION_H__
#define __IMAGE_CLASSIFICATION_H__

#include <stddef.h>
#include <stdint.h>

#include "include/ops/image_classification.h"
#include "include/ops/vaccel_ops.h"

struct vaccel_session;
struct vaccel_arg;

int vaccel_custom_image_classification_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
		int nr_read, struct vaccel_arg *write, int nr_write);

#endif 
```

```c
#include "image_classification.h"
#include "error.h"
#include "plugin.h"
#include "log.h"
#include "vaccel_ops.h"
#include "genop.h"

#include "session.h"

int vaccel_custom_image_classification(struct vaccel_session *sess, char* model_path,
		char* image_path, char* labels_path, float input_mean, float input_std,
		char* output)
{
	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%u Looking for plugin implementing VACCEL_IMG_CLASSIFICATION operation",
			sess->session_id);

	//Get implementation
	int (*plugin_op)() = get_plugin_op(VACCEL_IMG_CLASSIFICATION, sess->hint);
	if (!plugin_op)
		return VACCEL_ENOTSUP;

	return plugin_op(sess, model_path, image_path, labels_path,
			input_mean, input_std, output);
}

int vaccel_custom_image_classification_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
		int nr_read, struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 5) {
		vaccel_error("Wrong number of read arguments in VACCEL_IMG_CLASSIFICATION: %d",
				nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 1) {
		vaccel_error("Wrong number of write arguments in VACCEL_IMG_CLASSIFICATION: %d",
				nr_write);
		return VACCEL_EINVAL;
	}

	char* model_path  =  (char*)read[0].buf;
	char* image_path  =  (char*)read[1].buf;
	char* labels_path =  (char*)read[2].buf;
	float input_mean  =  *(float*)read[3].buf;
	float input_std   =  *(float*)read[4].buf;

	char* output      =  (char*)write[0].buf;

	return vaccel_custom_image_classification(sess, model_path, image_path, labels_path,
			input_mean, input_std, output);
}
```

## TPU-Plugin

## License

[Apache License 2.0](LICENSE)
