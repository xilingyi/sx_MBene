% This make.m is used under Windows

% mex -O -c svm.cpp
% mex -O -c svm_model_matlab.c
% mex -O svmtrain.c svm.obj svm_model_matlab.obj
% mex -O svmpredict.c svm.obj svm_model_matlab.obj
% mex -O libsvmread.c
% mex -O libsvmwrite.c

mex -largeArrayDims -c svm.cpp
mex -largeArrayDims -c svm_model_matlab.c
mex -largeArrayDims svmtrain.c svm.obj svm_model_matlab.obj
mex -largeArrayDims svmpredict.c svm.obj svm_model_matlab.obj
mex -largeArrayDims libsvmread.c
mex -largeArrayDims libsvmwrite.c
