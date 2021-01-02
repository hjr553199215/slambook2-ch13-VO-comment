add_executable(entry entry.cpp)
target_link_libraries(entry myslam ${THIRD_PARTY_LIBS} /usr/lib/x86_64-linux-gnu/libtiff.so.5)#这里不加三方库可以吗？可以试一下
