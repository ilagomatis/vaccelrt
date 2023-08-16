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
	//printf("model_path: %s\n", model_path);
	
	char *image_path = (char*)malloc(1000*sizeof(char));
	strcpy(image_path, "/home/mendel/classify/cat.rgb\0");
	//printf("image_path: %s\n", image_path);

	char *labels_path = (char*)malloc(1000*sizeof(char));
	strcpy(labels_path, "/home/mendel/classify/imagenet_labels.txt\0");
	//printf("labels_path: %s\n", labels_path);

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
