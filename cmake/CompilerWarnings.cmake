# LEARNER: Educational projects still benefit from pedantic warnings — they catch
# real bugs early. We apply warnings only to our own targets, not third-party code.

function(eduort_set_project_warnings target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "eduort_set_project_warnings: target '${target_name}' does not exist")
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target_name} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
      -Wnon-virtual-dtor
      -Wold-style-cast
      -Wcast-align
      -Wunused
      -Woverloaded-virtual
    )
    if(EDUORT_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
    if(EDUORT_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE /WX)
    endif()
  endif()
endfunction()
