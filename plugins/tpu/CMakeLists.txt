set(include_dirs ${CMAKE_SOURCE_DIR}/src)
set(SOURCES vaccel.c ${include_dirs}/vaccel.h ${include_dirs}/plugin.h)

add_library(vaccel-tpu SHARED ${SOURCES})
target_include_directories(vaccel-tpu PRIVATE ${include_dirs})

# Setup make install
install(TARGETS vaccel-tpu DESTINATION "${lib_path}")
