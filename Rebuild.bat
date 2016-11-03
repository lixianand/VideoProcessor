rmdir build /Q /S
mkdir build
cd build
cmake -G "MinGW Makefiles" -o3 ..
mingw32-make -o3