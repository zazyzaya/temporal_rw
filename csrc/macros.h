#pragma once

#ifdef _WIN32
#if defined(torchcluster_EXPORTS)
#define TEMPORAL_RW_API __declspec(dllexport)
#else
#define TEMPORAL_RW_API __declspec(dllimport)
#endif
#else
#define TEMPORAL_RW_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define TEMPORAL_RW_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define TEMPORAL_RW_INLINE_VARIABLE __declspec(selectany)
#else
#define TEMPORAL_RW_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
