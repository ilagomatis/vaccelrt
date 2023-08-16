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
	
	//printf("out: %s", out);
	
	//output = (char*)malloc(10000);
	strcpy(output, out);

	//printf("ouput: %s", output);	
	//output = out;

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
