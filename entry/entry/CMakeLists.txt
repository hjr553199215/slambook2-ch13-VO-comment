add_executable(entry entry.cpp)
target_link_libraries(entry myslam ${THIRD_PARTY_LIBS})#这里不加三方库可以吗？可以试一下