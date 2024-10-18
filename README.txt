Alan Wolfe:

Downloaded from https://abdallagafar.com/publications/gbn/

I had to jump through a couple hoops to get this to compile on windows, such as finding a getopt for windows and finding a compiled lib of cairo for windows.

The getopt I found doesn't work the same way as this code assumes. You need to put all positional arguments after all named arguments in the version i found.

Also, my compiled results had a problem with this code in gbn-adaptive.cu, which made it output white noise, not blue noise:

    GBNAdaptive gbn = (
        inputFileName ?
        GBNAdaptive(inputFileName, img) :
        GBNAdaptive(N, img, initRandom)
    );

The problem is that code makes temporary GBNAdaptive() objects for me, which allocate the buffers, and then copies all that work into gbn.
The temporary then goes out of scope which includes freeing the pointers it allocated.
Later on, cudaMemcpy's failed in GBNAdaptive::optimize() because they were trying to copy to and from freed host memory.

gbn-adaptive seems to work well with these changes. I haven't fully tested the others so I'm not sure if they need other changes to work correctly.

===================== ORIGINAL README BELOW =====================

- This code is provided without any guarantees; please review the code first.

- Please feel free to use it as suits your needs.

- Recognition of the author is highly appreciated.

- I provided a photo of myself for testing.

- Please check my website, abdallagafar.com, for updates.

- I am a researcher but not a professional programmer, so I am relying on
individual files to demonstrate the core algorithms. I may consider building a proper make utility in the future.

- Here is an example compilation line:

    nvcc -Xcompiler -O3,-march=native,-msse4.1 -o spectrum spectrum.cu -lcairo


and here are example run lines:

    ./gbn-adaptive taksim-circle.pgm 10000 1000 test.{pdf,png}

    ./gbn-bounded 1024 1000 test.txt && ./spectrum test.txt -o test.png -r test-rp.tex && pdflatex test-rp.tex

    ./gbn-toroidal 1024 1000 test.txt && ./spectrum test.txt -o test.png -r test-rp.tex && pdflatex test-rp.tex

    ./gbn-reconstruct -g 0.43 -w 512 -a 187 -i 15 test.txt test.png

