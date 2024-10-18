rem NEED to set the path to the 64 bit tools, else it will crash!
set path=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;%path%

rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o spectrum spectrum.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o gbn-toroidal gbn-toroidal.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o gbn-adaptive gbn-adaptive.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o gbn-reconstruct gbn-reconstruct.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o gbn-bounded gbn-bounded.cu -lcairo
rem nvcc -Xcompiler -O3,-march=native,-msse4.1 -o spectrum-nd spectrum-nd.cu -lcairo

nvcc -o spectrum spectrum.cu -lcairo
nvcc -o gbn-toroidal gbn-toroidal.cu -lcairo
nvcc -o gbn-adaptive gbn-adaptive.cu -lcairo
nvcc -o gbn-reconstruct gbn-reconstruct.cu -lcairo
nvcc -o gbn-bounded gbn-bounded.cu -lcairo
nvcc -o spectrum-nd spectrum-nd.cu -lcairo