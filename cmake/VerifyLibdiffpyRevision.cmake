# Source archives deliberately have no .git entry. Test for one before running
# Git so an archive unpacked inside another repository cannot inherit its HEAD.
function(verify_libdiffpy_revision source_dir expected_revision)
    if(NOT EXISTS "${source_dir}/.git")
        return()
    endif()

    find_package(Git REQUIRED)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --verify HEAD
        RESULT_VARIABLE git_result
        OUTPUT_VARIABLE actual_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE git_error
    )
    if(NOT git_result EQUAL 0)
        message(FATAL_ERROR "Cannot determine the libdiffpy revision: ${git_error}")
    endif()
    if(NOT actual_revision STREQUAL expected_revision)
        message(FATAL_ERROR
            "libdiffpy revision mismatch: cmake/LibdiffpyVersion.cmake records "
            "${expected_revision}, but ${source_dir} is at ${actual_revision}. "
            "Use a checkout of the pinned commit, or unset "
            "FETCHCONTENT_SOURCE_DIR_LIBDIFFPY to download the pinned archive."
        )
    endif()
endfunction()
