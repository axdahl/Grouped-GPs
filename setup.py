import os
import sys

#Compile the TensorFlow ops.
compile_command = ("g++ -std=c++11 -shared ./autogp/util/tf_ops/vec_to_tri.cc "
                   "./autogp/util/tf_ops/tri_to_vec.cc -o ./autogp/util/tf_ops/matpackops.so "
                   "-fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')")

#if sys.platform == "darwin":
#    compile_command += " -undefined dynamic_lookup"

#if sys.platform == "linux2":
#    compile_command += " -D_GLIBCXX_USE_CXX11_ABI=0 "

compile_command += " -D_GLIBCXX_USE_CXX11_ABI=0 "

os.system(compile_command)
