#include <math.h>
#include "mex.h"
typedef struct FCOMPLEX {double r,i;} fcomplex;
fcomplex Cadd(fcomplex a, fcomplex b)
{
	fcomplex c;
	c.r=a.r+b.r;
	c.i=a.i+b.i;
	return c;
}

fcomplex Csub(fcomplex a, fcomplex b)
{
	fcomplex c;
	c.r=a.r-b.r;
	c.i=a.i-b.i;
	return c;
}
fcomplex Cmul(fcomplex a, fcomplex b)
{
	fcomplex c;
	c.r=a.r*b.r-a.i*b.i;
	c.i=a.i*b.r+a.r*b.i;
	return c;
}

fcomplex Complex(double re, double im)
{
	fcomplex c;
	c.r=re;
	c.i=im;
	return c;
}

fcomplex Conjg(fcomplex z)
{
	fcomplex c;
	c.r=z.r;
	c.i = -z.i;
	return c;
}

fcomplex Cdiv(fcomplex a, fcomplex b)
{
	fcomplex c;
	float r,den;
	if (fabs(b.r) >= fabs(b.i)) {
		r=b.i/b.r;
		den=b.r+r*b.i;
		c.r=(a.r+r*a.i)/den;
		c.i=(a.i-r*a.r)/den;
	} else {
		r=b.r/b.i;
		den=b.i+r*b.r;
		c.r=(a.r*r+a.i)/den;
		c.i=(a.i*r-a.r)/den;
	}
	return c;
}

float Cabs(fcomplex z)
{
	float x,y,ans,temp;
	x=fabs(z.r);
	y=fabs(z.i);
	if (x == 0.0)
		ans=y;
	else if (y == 0.0)
		ans=x;
	else if (x > y) {
		temp=y/x;
		ans=x*sqrt(1.0+temp*temp);
	} else {
		temp=x/y;
		ans=y*sqrt(1.0+temp*temp);
	}
	return ans;
}

fcomplex Csqrt(fcomplex z)
{
	fcomplex c;
	float x,y,w,r;
	if ((z.r == 0.0) && (z.i == 0.0)) {
		c.r=0.0;
		c.i=0.0;
		return c;
	} else {
		x=fabs(z.r);
		y=fabs(z.i);
		if (x >= y) {
			r=y/x;
			w=sqrt(x)*sqrt(0.5*(1.0+sqrt(1.0+r*r)));
		} else {
			r=x/y;
			w=sqrt(y)*sqrt(0.5*(r+sqrt(1.0+r*r)));
		}
		if (z.r >= 0.0) {
			c.r=w;
			c.i=z.i/(2.0*w);
		} else {
			c.i=(z.i >= 0) ? w : -w;
			c.r=z.i/(2.0*c.i);
		}
		return c;
	}
}

fcomplex RCmul(double x, fcomplex a)
{
	fcomplex c;
	c.r=x*a.r;
	c.i=x*a.i;
	return c;
}
void quadsolve(mxArray const* A, mxArray const* B, mxArray const* C,
               mxArray * X1, mxArray * X2, mxArray * D,
               size_t N, size_t M)
{

    mwSize i,j;
    double  *Ar, *Ai, *Br, *Bi, *Cr, *Ci, *X1r, *X1i, *X2r, *X2i, *Dr, *Di;
    Ar = mxGetPr(A);
    Ai = mxGetPi(A);
    Br = mxGetPr(B);
    Bi = mxGetPi(B);
    Cr = mxGetPr(C);
    Ci = mxGetPi(C);
    X1r = mxGetPr(X1);
    X1i = mxGetPi(X1);
    X2r = mxGetPr(X2);
    X2i = mxGetPi(X2);
    Dr = mxGetPr(D);
    Di = mxGetPi(D);
    
    /* get pointers to the real and imaginary parts of the inputs */
    
    for(i=0; i<N; i++) {
        for(j=0; j<M; j++) {
            fcomplex a;
            fcomplex b;
            fcomplex c;
            if (mxIsComplex(A)){
                a.r = Ar[i*N+j];
                a.i = Ai[i*N+j];
            }else{
                a.r = Ar[i*N+j];
                a.i = 0;
            }
            if (mxIsComplex(B)){
                b.r = Br[i*N+j];
                b.i = Bi[i*N+j];
            }else{
                b.r = Br[i*N+j];
                b.i = 0;
            }
            if (mxIsComplex(C)){
                c.r = Cr[i*N+j];
                c.i = Ci[i*N+j];
            }else{
                c.r = Cr[i*N+j];
                c.i = 0;
            }
            
            fcomplex h_1 = Cmul(b,b);
            fcomplex h_2 = Cmul(a,c);
            fcomplex h_3 = {4,0};
            h_3 = Cmul(h_3,h_2);
            fcomplex d = Csub(h_1,h_3);
            Dr[i*N+j] = d.r;
            Di[i*N+j] = d.i;
            fcomplex h_4 = Csqrt(d);
            fcomplex h_5 = {2,0};
            fcomplex h_6 = {-2,0};
            fcomplex x1 = Csub(h_4,b);
            fcomplex x2 = Cadd(b,h_4);
            x1 = Cdiv(x1,a);
            x1 = Cdiv(x1,h_5);
            x2 = Cdiv(x2,a);
            x2 = Cdiv(x2,h_6);
            X1r[i*N+j] = x1.r;
            X1i[i*N+j] = x1.i;
            X2r[i*N+j] = x2.r;
            X2i[i*N+j] = x2.i;
            
        }
    }
     
}
void mexFunction(int nlhs, mxArray *plhs[], // результаты
                 int nrhs, const mxArray *prhs[]) // аргументы
{
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Three inputs required.");
    }
    if (nlhs<2 || nlhs>3){
        mexErrMsgIdAndTxt("MyToolbox:quadsolve:nlhs","Two or Three output required");
    }
    if( !mxIsNumeric(prhs[0]) || !mxIsNumeric(prhs[1]) || !mxIsNumeric(prhs[2]) ) {
        mexErrMsgIdAndTxt( "MyToolbox:quadsolve:plhs", "Inputs must be number.\n");
    }
    size_t rows, cols;
    /* get the length of each input vector */
    rows = mxGetN(prhs[0]);
    cols = mxGetM(prhs[0]);
    if (mxGetN(prhs[1])!=rows || mxGetN(prhs[2])!=rows){
        mexErrMsgIdAndTxt( "MyToolbox:quadsolve:plhs", "Sizes N must be equal.\n");
    }
    if (mxGetM(prhs[1])!=cols || mxGetM(prhs[2])!=cols){
        mexErrMsgIdAndTxt( "MyToolbox:quadsolve:plhs", "Sizes M must be equal.\n");
    }
    mxArray *plhs_1,*plhs_2,*plhs_3;
    plhs_1 = mxCreateDoubleMatrix( (mwSize)rows, (mwSize)cols, mxCOMPLEX); // x1
    plhs_2 = mxCreateDoubleMatrix( (mwSize)rows, (mwSize)cols, mxCOMPLEX); // x2
    plhs_3 = mxCreateDoubleMatrix( (mwSize)rows, (mwSize)cols, mxCOMPLEX); // D
    
    
    quadsolve(prhs[0],prhs[1],prhs[2],plhs_1,plhs_2,plhs_3,rows,cols);
    int flag = 1;
    double  * x1 = mxGetPi(plhs_1);
    double  * x2 = mxGetPi(plhs_2);
    double  * D = mxGetPi(plhs_3);
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            if (x1[i*rows+j]>0.00001 || x1[i*rows+j]< -0.00001){
                flag=0;
            }
            if (x2[i*rows+j]>0.00001 || x2[i*rows+j]< -0.00001){
                flag=0;
            }
            if (D[i*rows+j]>0.00001 || D[i*rows+j]< -0.00001){
                flag=0;
            }
        }
    }
    double  * x1_r_solve = mxGetPr(plhs_1);
    double  * x2_r_solve = mxGetPr(plhs_2);
    double  * D_r_solve = mxGetPr(plhs_3);


    mexPrintf("%d",flag);
    if (flag==0){
        plhs[0] = plhs_1; // x1
        plhs[1] = plhs_2; // x2
        plhs[2] = plhs_3; // D
    }else{
        plhs[0] = mxCreateDoubleMatrix( (mwSize)rows, (mwSize)cols, mxREAL); // x1
        plhs[1] = mxCreateDoubleMatrix( (mwSize)rows, (mwSize)cols, mxREAL); // x2
        plhs[2] = mxCreateDoubleMatrix( (mwSize)rows, (mwSize)cols, mxREAL); // D
        double  * x1_ = mxGetPr(plhs[0]);
        double  * x2_ = mxGetPr(plhs[1]);
        double  * D_ = mxGetPr(plhs[2]);
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                x1_[i*rows+j] = x1_r_solve[i*rows+j];
                x2_[i*rows+j] = x2_r_solve[i*rows+j];
                D_[i*rows+j] = D_r_solve[i*rows+j];
            }
        }

    }
}
