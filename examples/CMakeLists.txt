# Examples — each subdirectory builds one standalone binary.
# cli/ is built from the root CMakeLists.txt (PARAKEET_BUILD_CLI).

add_subdirectory(basic)
add_subdirectory(timestamps)
add_subdirectory(beam-search)
add_subdirectory(phrase-boost)
add_subdirectory(batch)
add_subdirectory(vad)
add_subdirectory(gpu)
add_subdirectory(stream)
add_subdirectory(nemotron)
add_subdirectory(diarize)
add_subdirectory(diarized-transcription)
add_subdirectory(c-api)
