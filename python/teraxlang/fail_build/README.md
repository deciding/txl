# xyzlang

```
pip install -r requirements.txt # torch must be installed before hand

cp zlang_ir.h
cp ir.cc
# CMAKELists.txt

cd thirdparty/triton

pip install -r python/requirements.txt
pip install -e python

export PYTHONPATH=$(pwd)/xyzlang/

# careful if you have conda with other versions of compiler
# printenv | grep x86_64-conda-linux-gnu-c
# export CC=/usr/bin/cc
# export CXX=/usr/bin/c++

pip install -e xyzlang/zlang/
#TODO cp libzlang.so

TRITON_ALWAYS_COMPILE=1 python xyzlang/zlang/tests/01-vector-add.py
```
