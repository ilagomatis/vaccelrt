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
	int (*plugin_op)() = get_plugin_op(VACCEL_IMG_CLASSIFICATION);
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

	char* model_path  =  *(char**)read[0].buf;
	char* image_path  =  *(char**)read[1].buf;
	char* labels_path =  *(char**)read[2].buf;
	float input_mean  =  *(float*)read[3].buf;
	float input_std   =  *(float*)read[4].buf;

	char* output      =  *(char**)write[0].buf;

	return vaccel_custom_image_classification(sess, model_path, image_path, labels_path,
			input_mean, input_std, output);
}
