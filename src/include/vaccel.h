/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __VACCEL_H__
#define __VACCEL_H__

#define VACCELRT_VERSION "fcbde9d-dirty"

#include "error.h"
#include "log.h"
#include "plugin.h"
#include "session.h"
#include "tf_model.h"
#include "vaccel_id.h"
#include "ops/vaccel_ops.h"
#include "ops/blas.h"
#include "ops/minmax.h"
#include "ops/exec.h"
#include "ops/genop.h"
#include "ops/image.h"
#include "ops/noop.h"
#include "ops/tf.h"
#include "ops/fpga.h"
#include "ops/image_classification.h"
#include "resources/tf_saved_model.h"
#include "resources/shared_object.h"
#include "misc.h"
#include "vaccel_prof.h"
#include "ops/torch.h"
#include "resources/torch_saved_model.h"
#endif /* __VACCEL_H__ */
