# gpt2-scratch
A simple c++ implementation of GPT2

### Dependencies
regular expression lib: https://github.com/google/re2 

### Build on Windows
###### build with MinGW (re2 installed by vcpkg)
```bash
vcpkg install re2:x64-mingw-static

cmake . -B build -G "MinGW Makefiles"
```
