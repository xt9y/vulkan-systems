# Mbed TLS 3.6.7 uses a required pinned gitlink for "framework".
# c build exports source files from a bare Git cache without .git metadata,
# so regular "git submodule update --init" cannot run here.
# This hook executes during Mbed TLS project(), before add_subdirectory(framework).
set(_framework_ref "dde0c4a0e448a0552f18817dcea633bb851fd288")
set(_framework_url "https://github.com/Mbed-TLS/mbedtls-framework.git")
set(_framework_dir "${CMAKE_CURRENT_SOURCE_DIR}/framework")
find_program(_framework_git NAMES git REQUIRED)
if(NOT EXISTS "${_framework_dir}/CMakeLists.txt")
    if(EXISTS "${_framework_dir}")
        file(REMOVE_RECURSE "${_framework_dir}")
    endif()
    execute_process(
        COMMAND "${_framework_git}" init "${_framework_dir}"
        RESULT_VARIABLE _init_rc ERROR_VARIABLE _init_err)
    if(NOT _init_rc EQUAL 0)
        message(FATAL_ERROR "Mbed TLS framework init failed: ${_init_err}")
    endif()
    execute_process(
        COMMAND "${_framework_git}" -C "${_framework_dir}" remote add origin "${_framework_url}"
        RESULT_VARIABLE _remote_rc ERROR_VARIABLE _remote_err)
    if(NOT _remote_rc EQUAL 0)
        message(FATAL_ERROR "Mbed TLS framework remote failed: ${_remote_err}")
    endif()
    execute_process(
        COMMAND "${_framework_git}" -C "${_framework_dir}" fetch --depth=1 origin "${_framework_ref}"
        RESULT_VARIABLE _fetch_rc ERROR_VARIABLE _fetch_err)
    if(NOT _fetch_rc EQUAL 0)
        message(FATAL_ERROR "Pinned Mbed TLS framework fetch failed: ${_fetch_err}")
    endif()
    execute_process(
        COMMAND "${_framework_git}" -C "${_framework_dir}" checkout --detach "${_framework_ref}"
        RESULT_VARIABLE _checkout_rc ERROR_VARIABLE _checkout_err)
    if(NOT _checkout_rc EQUAL 0)
        message(FATAL_ERROR "Pinned Mbed TLS framework checkout failed: ${_checkout_err}")
    endif()
endif()
# The commit must exactly match Mbed TLS's gitlink, even if already populated.
execute_process(
    COMMAND "${_framework_git}" -C "${_framework_dir}" rev-parse HEAD
    OUTPUT_VARIABLE _framework_actual
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _verify_rc ERROR_VARIABLE _verify_err)
if(NOT _verify_rc EQUAL 0 OR
   NOT _framework_actual STREQUAL _framework_ref OR
   NOT EXISTS "${_framework_dir}/CMakeLists.txt")
    message(FATAL_ERROR "Mbed TLS framework revision mismatch; expected ${_framework_ref}. ${_verify_err}")
endif()
