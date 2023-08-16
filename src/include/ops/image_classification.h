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
