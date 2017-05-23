function b = subsref(a,index)
%SUBSREF Define field name indexing for multinomial objects

% Copyright October, 2006, Brown University, Providence, RI. 
% All Rights Reserved 

% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a commercial
% product is hereby granted without fee, provided that the above copyright
% notice appear in all copies and that both that copyright notice and this
% permission notice appear in supporting documentation, and that the name of
% Brown University not be used in advertising or publicity pertaining to
% distribution of the software without specific, written prior permission. 

% BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE. IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY
% SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
% RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
% CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
% CONNECTION WITH THE USE.

% Author: Frank Wood fwood@cs.brown.edu

switch index.type
case '()'
    b = a.ps(index.subs{:});
    case '.'
        if(strcmp(index.subs,'covariance'))
            b = a.c;
        elseif(strcmp(index.subs,'mean'))
            b = a.m;
        end
    otherwise
    error('Cell and name indexing not supported by Gaussian objects')
end
