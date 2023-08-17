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

vaccelrt/src/include/ops/image_classification.h
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

vaccelrt/src/ops/image_classification.h
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

vaccelrt/src/ops/image_classification.c
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
vaccelrt/plugins/tpu/vaccel.c
```c
#include <stdio.h>
#include <plugin.h>
#include "ops/vaccel_ops.h"
#include "classify.h"
#include <ops/image_classification.h>

static int tpu_custom_image_classification(struct vaccel_session *session,
					   char* model_path,
                		       	   char* image_path,
                		           char* labels_path,
                		           float input_mean,
                		           float input_std,
	       			           char* output)
{
	fprintf(stdout, "Calling tpu-image-classification for session %u\n", session->session_id);
	printf("---\n\n");

	printf("model_path: %s\n", model_path);
	printf("image_path: %s\n", image_path);
	printf("labels_path: %s\n", labels_path);

	char*  out = classify_image(model_path,
			       	          image_path,
					  labels_path,
					  input_mean,
					  input_std);
	
	strcpy(output, out);

	printf("\n\n---\n\n");
	printf("Ending custom-image-classification operation here");

	return VACCEL_OK;
}

struct vaccel_op op = VACCEL_OP_INIT(op, VACCEL_IMG_CLASSIFICATION, tpu_custom_image_classification);

static int init(void)
{
	return register_plugin_function(&op);
}

static int fini(void)
{
	return VACCEL_OK;
}

VACCEL_MODULE(
	.name = "tpu_custom_image_classification",
	.version = "0.1",
	.init = init,
	.fini = fini
)
```
vaccelrt/plugins/tpu/classify.h
```c
#ifdef __cplusplus
extern "C" {
#endif

char* classify_image(
                char* model_path,
                char* image_path,
                char* labels_path,
                float input_mean,
                float input_std
            );

#ifdef __cplusplus
}
#endif
```

## Example
vaccelrt/examples/tpu_img_class.c
```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vaccel.h>

int main()
{
	int ret;
	struct vaccel_session sess;
	
	char *model_path = (char*)malloc(1000*sizeof(char));
	strcpy(model_path, "/home/mendel/classify/mobilenet_v1_1.0_224_quant_edgetpu.tflite\0");
	
	char *image_path = (char*)malloc(1000*sizeof(char));
	strcpy(image_path, "/home/mendel/classify/cat.rgb\0");

	char *labels_path = (char*)malloc(1000*sizeof(char));
	strcpy(labels_path, "/home/mendel/classify/imagenet_labels.txt\0");

	float input_mean = 128;
	float input_std = 128;
	
	char* output = (char*)malloc(1000*sizeof(char));

	ret = vaccel_sess_init(&sess, 0);
	if (ret != VACCEL_OK) {
		fprintf(stderr, "Could not initialize session\n");
		return 1;
	}


	printf("Initialized session with id: %u\n", sess.session_id);


	ret = vaccel_custom_image_classification(&sess, model_path, image_path, labels_path, input_mean, input_std, output);
	if (ret) {
		fprintf(stderr, "Could not run op: %d\n", ret);
		goto close_session;
	}

	printf("\nOutput from tpu-plugin: \n%s\n", output);


close_session:
	if (vaccel_sess_free(&sess) != VACCEL_OK) {
		fprintf(stderr, "Could not clear session\n");
		return 1;
	}
	
	free(model_path);
	free(image_path);
	free(labels_path);
	free(output);

	return ret;
}
```
## License

[Apache License 2.0](LICENSE)
