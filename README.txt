README

In order to run all three methods, run GPregression.m.
Note 1: this will take a while, but will print intermediate time stamps

Note 2: the Chebfun package, and Symbolic math toolbox is required to run GPregression.m.
https://www.chebfun.org/

Note 3: GPregression.m returns two plots showing the mean function for n=2^16 and 2^20.
	Also, returns a plot showing the run-time for all three methods.
---------------------------------------------------------------
<Descriptions>

HODLR.m: function for posterior mean via HODLR factorization 

KLexpansion.m: function for global low-rank factors via KL-expansion 
		(requires Chebfun and Symbolic math toolbox)

REig.m: function for global low-rank factors via single-pass  Hermitian Randomized SVD.


KLLR.m: Low rank factors via KL-expansion for off-diagonal low-rank matrices.


PQR.m: function that returns low-rank factors U,V such that
	A \approx U*V.
	Uses deterministic method (rank-revealing pivoted qr)
	input is a matrix A, and desired rank r (typically r=15).
	Call is [U,V] = PQR(A, r)

RLR.m: function that returns low-rank factors U,V such that
	A \approx U*V.
	Uses randomized method
	input is a matrix A, and desired rank r and over-sampling parameter p.
	(typically r=15, p=10).
	Call is [U,V] = PQR(A, r, p)
