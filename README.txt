README

The Chebfun package, and Symbolic math toolbox is required to call the function
KLexpansion.m, which is called in GPregression.m.
https://www.chebfun.org/


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
