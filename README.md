# gpt2-scratch
A simple c++ implementation of GPT2

### Dependencies
regular expression lib: https://github.com/google/re2 

### Build on Windows
###### build with MinGW (re2 installed by vcpkg)
```bash
cmake . -B build -G "MinGW Makefiles"  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake   -DVCPKG_TARGET_TRIPLET=x64-mingw-static
```
