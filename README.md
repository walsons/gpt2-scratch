# gpt2-scratch
A simple c++ implementation of GPT2

### Dependencies
regular expression lib: https://github.com/google/re2 

### Build on Windows
###### build with MinGW (re2 installed by vcpkg)
```bash
#Before building, run custom_state_dict.py to generate the custom model parameter file and place it in the assets directory.

vcpkg install re2:x64-mingw-static

cmake . -B build -G "MinGW Makefiles"
```
