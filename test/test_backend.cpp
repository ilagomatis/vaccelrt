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

#include "gtest/gtest.h"

extern "C" {
#include "plugin.h"
#include "error.h"
#include "list.h"
}

#include <stdbool.h>
#include <string.h>

static const char *pname = "mock_plugin";

class PluginTests : public ::testing::Test {
	static int fini(void)
	{
		return VACCEL_OK;
	} 
	static int init(void)
	{
		return VACCEL_OK;
	}

protected:
	/* initialize structures with zeros */
	struct vaccel_plugin plugin = {0};
	struct vaccel_plugin_info pinfo = {0};

	void SetUp() override
	{
		plugin.info = &this->pinfo;
		plugin.info->name = pname;
		list_init_entry(&plugin.entry);
		list_init_entry(&plugin.ops);
		plugin.info->init = init;
		plugin.info->fini = fini;

		plugins_bootstrap();
	}

	void TearDown() override
	{
		plugins_shutdown();
	}
};

TEST_F(PluginTests, plugin_init_null) {
	ASSERT_EQ(register_plugin(NULL), VACCEL_EINVAL);
}

TEST_F(PluginTests, plugin_init_null_name) {
	struct vaccel_plugin_info info = { 0 };
	struct vaccel_plugin plugin;
	plugin.info = &info;
	plugin.entry = LIST_ENTRY_INIT(plugin.entry);

	ASSERT_EQ(register_plugin(&plugin), VACCEL_EINVAL);
}

TEST(plugin_init_not_bootstraped, not_ok) {
	struct vaccel_plugin plugin = {};
	ASSERT_EQ(register_plugin(&plugin), VACCEL_EBACKEND);
}

TEST_F(PluginTests, plugin_init_values) {
	ASSERT_EQ(register_plugin(&plugin), VACCEL_OK);
	ASSERT_EQ(strcmp(plugin.info->name, "mock_plugin"), 0);
	ASSERT_TRUE(list_empty(&plugin.ops));
}

TEST_F(PluginTests, plugin_register_null) {
	ASSERT_EQ(register_plugin(NULL), VACCEL_EINVAL);
}

TEST(plugin_register_not_bootstraped, not_ok) {
	struct vaccel_plugin plugin = {};
	ASSERT_EQ(register_plugin(&plugin), VACCEL_EBACKEND);
}

TEST_F(PluginTests, plugin_register_existing) {
	ASSERT_EQ(register_plugin(&plugin), VACCEL_OK);
	ASSERT_EQ(register_plugin(&plugin), VACCEL_EEXISTS);
}

TEST_F(PluginTests, plugin_register_with_ops) {
}
