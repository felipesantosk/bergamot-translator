


add_executable(DistilledModel distilled.cpp)
target_link_libraries(DistilledModel dl marian ssplit)


add_executable(DistilledSaveFile save_file.cpp)
target_link_libraries(DistilledSaveFile dl marian ssplit)


