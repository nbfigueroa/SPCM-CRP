This software package implements the connectivity clustering method
described in C. Baldassano, D.M. Beck, L. Fei-Fei. "Parcellating
connectivity in spatial maps." PeerJ, 2015. Please refer to the journal
article for mathematical details, and please cite this article if you
publish results using this software.

MATLAB:
A demo of the method on synthetic data can be run with the commands
>> matlabpool open 10
>> [WC DC DC_K] = LearnSynth('stripes');
>> plot(0:9,mean(WC,2),0:9,mean(DC,2))
This will plot the performance of our dd-CRP based method compared to
Ward clustering, as in Figure 2 of the paper.
The code was developed under MATLAB version 7.13.0.564 (R2011b), but should
run in any recent MATLAB version. It does not rely on any external libraries.

python:
A demo of the method on synthetic data can be run with the command
python LearnSynth.py
which will print out the NMI performance of Ward clustering versus our
dd-CRP clustering, at 10 different noise levels, as in Figure 2 of the paper.
This code was developed under Python 3.4.1 using Anaconda 2.0.1 (64-bit),
but should run in any recent python distribution with the numpy and scipy
packages.

Please send questions or bug reports to Chris Baldassano, <chrisb33@cs.stanford.edu>



Copyright (c) 2015, Christopher Baldassano and Henry Jung, Stanford University
All rights reserved.
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Stanford University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL CHRISTOPHER BALDASSANO, HENRY JUNG, OR
STANFORD UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.