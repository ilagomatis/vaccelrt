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
