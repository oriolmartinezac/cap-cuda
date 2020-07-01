/*
 * nn.c
 *
 *  Created on: 5 jul. 2016
 *  Author: ecesar
 *
 *      Descripció:
 *      Xarxa neuronal simple de tres capes. La d'entrada que són els pixels d'una
 *      imatge (mirar descripció del format al comentari de readImg) de 32x32 (un total de 1024
 *      entrades). La capa oculta amb un nombre variable de neurones (amb l'exemple proporcionat 117
 *      funciona relativament bé, però si incrementem el nombre de patrons d'entrament caldrà variar-lo).
 *      Finalment, la capa de sortida (que ara té 10 neurones ja que l'entrenem per reconéixer 10
 *      patrons ['0'..'9']).
 *      El programa passa per una fase d'entrenament en la qual processa un conjunt de patrons (en
 *      l'exemple proporcionat són 1934 amb els dígits '0'..'9', escrits a mà). Un cop ha calculat
 * 	    els pesos entre la capa d'entrada i l'oculta i entre
 *      aquesta i la de sortida, passa a la fase de reconèixament, on llegeix 946 patrons d'entrada
 *      (es proporcionen exemples per aquests patrons), i intenta reconèixer de quin dígit es tracta.
 *
 *  Darrera modificació: gener 2019. Ara l'aprenentatge fa servir la tècnica dels mini-batches
 */

/*******************************************************************************
*    Aquest programa és una adaptació del fet per  JOHN BULLINARIA
*    ( http://www.cs.bham.ac.uk/~jxb/NN/nn.html):
*
*    nn.c   1.0                                       � JOHN BULLINARIA  2004  *
*******************************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <limits.h>
#include "common.h"
int total;
int seed=50;

#define block_sizeIH 32;
#define block_sizeHO 32;

inline int rando()
{
    seed = (214013*seed+2531011);
    return seed>>16;
}

inline float frando()
{
    return (rando()/65536.0f);
}

void freeDeltaWeights(float *DeltaWeightIH[], float *DeltaWeightHO[]){
	for( int i = 0; i < NUMHID; i++)
		free(DeltaWeightIH[i]);
	for( int i = 0; i < NUMOUT; i++)
		free(DeltaWeightHO[i]);
}

void freeWeights(float *WeightIH[],  float *WeightHO[]){
	for( int i = 0; i < NUMHID; i++)
		free(WeightIH[i]);
	for( int i = 0; i < NUMOUT; i++)
		free(WeightHO[i]);
}

void freeTSet( int np, char **tset ){
	for( int i = 0; i < np; i++ ) free( tset[i] );
	free(tset);
}


__global__ void Kernel_for_1( float *d_WeightIH, char *d_tSet, int p, float *d_SumH)
{
	__shared__ float totalSumH[NUMIN]; //threads per block

	unsigned int indexi = threadIdx.x; //index per cada thread
	unsigned int indexj = blockIdx.x; //ID del block

	totalSumH[indexi] = d_WeightIH[indexj*NUMIN+indexi] * d_tSet[p*NUMIN+indexi];

	for(int stride = NUMIN/2; stride > 0; stride /=2 )
	{
		__syncthreads();
		if(indexi < stride ) totalSumH[indexi] += totalSumH[indexi + stride];
	}

	if(indexi == 0)
		d_SumH[indexj] = totalSumH[0];
}

__global__ void Kernel_SumH(float *d_SumH, float *d_Hidden, int p)
{
	unsigned int indexi = threadIdx.x;
	unsigned int indexj = blockIdx.x;

	d_Hidden[p*NUMHID+indexi] = 1.0/(1.0 + expf(-d_SumH[indexi]));
}

//Kernel_for_2<<<NUMOUT, NUMHID>>>(d_WeightHO, d_Hidden, d_Target, d_Output, d_DeltaO, p, d_BError);
__global__ void Kernel_for_2(float *d_WeightHO,float *d_Hidden, float *d_Target,float *d_Output, float *d_DeltaO,int p, float *d_BError)
{
	__shared__ float totalSumO[NUMHID]; // no es potencia de 2 (117) suma + 11

	unsigned int indexi = threadIdx.x; //index per cada thread
	unsigned int indexj = blockIdx.x; //ID del block

	unsigned int pow = 1;
	while (pow < NUMHID) pow *= 2;

	totalSumO[indexi] = d_Hidden[p*NUMHID+indexi] * d_WeightHO[indexj*NUMHID+indexi];

	for(int stride = pow/2; stride > 0; stride /=2 )
	{
		__syncthreads();
		if(indexi < stride)
			if((indexi+stride)<NUMHID)
				totalSumO[indexi] += totalSumO[indexi + stride];
	}

	if (indexi == 0)
	{
		d_Output[p*NUMOUT+indexj] = 1.0/(1.0 + expf(-totalSumO[0]));
		d_BError[indexj] = 0.5 * (d_Target[p*NUMOUT+indexj] - d_Output[p*NUMOUT+indexj]) * (d_Target[p*NUMOUT+indexj] - d_Output[p*NUMOUT+indexj]);
		d_DeltaO[indexj] = (d_Target[p*NUMOUT+indexj] - d_Output[p*NUMOUT+indexj]) * d_Output[p*NUMOUT+indexj] * (1.0 - d_Output[p*NUMOUT+indexj]);   // Sigmoidal Outputs, SSE
	}
}

__global__ void Kernel_BError(float *d_BError, float *d_BErrorAcc)  //Reduction BError (PASSAR NUMOUT)
{
	__shared__ float totalBError[NUMOUT]; //multiple de 2

	unsigned int indexi = threadIdx.x;
	unsigned int indexj = blockIdx.x;

	totalBError[indexi] = d_BError[indexi];

	int pow = 1;
	while (pow < NUMOUT) pow *= 2;

	for(int stride = pow/2; stride > 0; stride /=2 )
	{
		__syncthreads();
		if(indexi < stride )
			if((indexi+stride) < NUMOUT)
				totalBError[indexi] += totalBError[indexi + stride];
	}

	if(indexi == 0)
	{
		totalBError[0] += d_BErrorAcc[0];
		d_BError[0] = totalBError[0];
		d_BErrorAcc[0] = d_BError[0];
	}

}

//Kernel_for_3<<<NUMHID, NUMOUT>>>(d_Hidden, d_DeltaH, p, WeightHO, DeltaO);
__global__ void Kernel_for_3(float *d_Hidden, float *d_DeltaH, int p, float *d_WeightHO, float *d_DeltaO)
{
	__shared__ float totalSumDOW[NUMOUT];

	unsigned int indexi = threadIdx.x;
	unsigned int indexj = blockIdx.x;

	int pow = 1;
	while (pow < NUMOUT) pow *= 2;

	totalSumDOW[indexi] = d_WeightHO[indexi*NUMHID+indexj] * d_DeltaO[indexi];

	for(int stride = pow/2; stride > 0; stride /=2 )
	{
		__syncthreads();
		if(indexi < stride )
			if((indexi+stride) < NUMOUT)
				totalSumDOW[indexi] += totalSumDOW[indexi + stride];
	}

	if(indexi == 0)
		d_DeltaH[indexj] = totalSumDOW[0] * d_Hidden[p*NUMHID+indexj] * (1.0 - d_Hidden[p*NUMHID+indexj]);
}

//Kernel_for_3_2<<<NUMHID, NUMIN>>>(d_DeltaH, p, d_DeltaWeightIH, eta, alpha, d_tSet);
__global__ void Kernel_for_3_2(float *d_DeltaH, int p,float *d_DeltaWeightIH, float eta, float alpha, char *d_tSet)
{
	unsigned int indexi = threadIdx.x;
	unsigned int indexj = blockIdx.x;
	unsigned int blockdim = blockDim.x;

	d_DeltaWeightIH[indexj*blockdim + indexi] =  eta * d_tSet[p*blockdim + indexi] * d_DeltaH[indexj] + alpha * d_DeltaWeightIH[indexj*blockdim + indexi];
}

//Kernel_for_4<<NUMOUT, NUMHID>>>(d_Hidden, d_DeltaO, p, d_DeltaWeightHO, eta, alpha);
__global__ void Kernel_for_4(float *d_Hidden, float *d_DeltaO, int p, float *d_DeltaWeightHO, float eta, float alpha)
{
	int indexi = threadIdx.x;
	int indexj = blockIdx.x;
	int blockdim = blockDim.x;

	d_DeltaWeightHO[indexj*blockdim + indexi] =  eta * d_Hidden[p*blockdim + indexi] * d_DeltaO[indexj] + alpha * d_DeltaWeightHO[indexj*blockdim + indexi];
}

__global__ void Kernel_for_WeightIH(float *d_WeightIH, float *d_DeltaWeightIH)
{
	unsigned int indexi = threadIdx.x;
	unsigned int indexj = blockIdx.x;

	d_WeightIH[indexj*NUMIN+indexi] += d_DeltaWeightIH[indexj*NUMIN+indexi];
}

/*
__global__ void Kernel_for_WeightIH(float *d_WeightIH, float *d_DeltaWeightIH)
{
	//unsigned int block_sizeIH = 32;
	__shared__ float ds_WeightIH[32][32];
	__shared__ float ds_DeltaWeightIH[32][32];

	unsigned int indexi = threadIdx.x;
	unsigned int indexj = blockIdx.x;

	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	unsigned int row = by* + ty;
	unsigned int col = bx*32 + tx;

	//LOAD MATRIX TO SHARED AND OPERATE
	for(int m = 0; m < NUMIN/32; m++)
	{
		//LOAD
		ds_WeightIH[ty][tx] = d_WeightIH[row*NUMIN + m*32 + tx];
		ds_DeltaWeightIH[ty][tx] = d_DeltaWeightIH[row*NUMIN + m*32 + tx];
		//__syncthreads();

			//WeightIH[j][i] += DeltaWeightIH[j][i];
		ds_WeightIH[ty][tx] += ds_DeltaWeightIH[ty][tx];
		__syncthreads();

		d_WeightIH[row*NUMIN + m*32 + tx] = ds_WeightIH[ty][tx];

		__syncthreads();
	}
}
*/
__global__ void Kernel_for_WeightHO(float *d_WeightHO, float *d_DeltaWeightHO)
{
	unsigned int indexi = threadIdx.x;
	unsigned int indexj = blockIdx.x;

	d_WeightHO[indexj*NUMHID+indexi] += d_WeightHO[indexj*NUMHID+indexi];
}
/*
__global__ void Kernel_for_WeightHO(float *d_WeightHO, float *d_DeltaWeightHO)
{
	//unsigned int block_sizeHO = 32;
	__shared__ float ds_WeightHO[32][32];
	__shared__ float ds_DeltaWeightHO[32][32];

	unsigned int indexi = threadIdx.x;
	unsigned int indexj = blockIdx.x;

	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	unsigned int row = by*32 + ty;
	unsigned int col = bx*32 + tx;

	//LOAD MATRIX TO SHARED AND OPERATE
	for(int m = 0; m < NUMOUT/32; m++)
	{
		//LOAD
		ds_WeightHO[ty][tx] = d_WeightHO[row*NUMOUT + m*32 + tx];
		ds_DeltaWeightHO[ty][tx] = d_DeltaWeightHO[row*NUMOUT + m*32 + tx];

			//WeightIH[j][i] += DeltaWeightIH[j][i];
		ds_WeightHO[ty][tx] += ds_DeltaWeightHO[ty][tx];
		__syncthreads();

		d_WeightHO[row*NUMOUT + m*32 + tx] = ds_WeightHO[ty][tx];

		__syncthreads();
	}
}
*/

void trainN()
{

	char tSet[NUMPAT][NUMIN];

	if( (loadPatternSet(tSet, (char *)"optdigits.tra", 1)) == 0){
        printf("Loading Patterns: Error!!\n");
		exit(-1);
	}

		float DeltaWeightIH[NUMHID][NUMIN];
    	float smallwt = 0.22;
/*
	for( int i = 0; i < NUMHID; i++){

		if ((WeightIH[i] = (double *)malloc((NUMIN)*sizeof(double))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}

		if ((DeltaWeightIH[i] = (float *)malloc((NUMIN)*sizeof(float))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}

	}
	*/
	for(int j = 0; j < NUMIN; j++)
		for( int i = 0; i < NUMHID; i++){
			WeightIH[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
			//DeltaWeightIH[i][j] = 0.0;
		}

	float DeltaWeightHO[NUMOUT][NUMHID];

	/*
	for( int i = 0; i < NUMOUT; i++){

		if ((WeightHO[i] = (float *)malloc((NUMHID)*sizeof(float))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}

		if ((DeltaWeightHO[i] = (float *)malloc((NUMHID)*sizeof(float))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}
	}
	*/
	for(int j = 0; j < NUMHID; j++)
		for( int i = 0; i < NUMOUT; i++){
			WeightHO[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
			//DeltaWeightHO[i][j] = 0.0;
		}

	float Error, BError, eta = 0.3, alpha = 0.5;
	int ranpat[NUMPAT];

	float Hidden[NUMPAT][NUMHID];

	float Output[NUMPAT][NUMOUT], DeltaO[NUMOUT], DeltaH[NUMHID];

	//float BError[NUMOUT];

	//BEGIN:Declaració matrius DEVICE

	   /*
	//32x32 per block
	int block_sizeIH = 32;
	int nBlocksXIH = ceil(NUMIN/block_sizeIH); //64
	int nBlocksYIH = ceil(NUMHID/block_sizeIH); //8

	dim3 blocksIH( block_sizeIH, block_sizeIH);
	dim3 gridIH(nBlocksXIH, nBlocksYIH);


	//8x8 per block
	int block_sizeIH = 8;
	int nBlocksXHO = ceil(NUMOUT/block_sizeIH); //64
	int nBlocksYHO = ceil(NUMHID/block_sizeIH); //8

	dim3 blocksHO (block_sizeHO, block_sizeHO);
	dim3 gridHO (nBlocksXHO, nBlocksYHO);
	*/
	//FOR 1
	float *d_WeightIH;
	float *d_Hidden;
	char *d_tSet;
	float *d_totalSumH;
	float *d_SumH;

	//Assignar espai en la memoria de la grafica
	cudaMalloc((void **)&d_WeightIH, NUMHID*NUMIN*sizeof(float));
	cudaMalloc((void **)&d_Hidden, NUMPAT*NUMHID*sizeof(float));
	cudaMalloc((void **)&d_tSet, NUMPAT*NUMIN*sizeof(char));
	cudaMalloc((void **)&d_totalSumH, 1*sizeof(float));
	cudaMalloc((void **)&d_SumH, NUMHID*sizeof(float));
	//Copiar Host to Device
	cudaMemcpy(d_WeightIH, WeightIH, NUMHID*NUMIN*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tSet, tSet, NUMPAT*NUMIN*sizeof(char), cudaMemcpyHostToDevice);

	//FOR 2
	float *d_Target;
	float *d_Output;
	float *d_DeltaO;
	float *d_BError;
	float *d_BErrorAcc;
	float *d_WeightHO;

	//Assignar espai en la memoria de la grafica
	cudaMalloc((void **)&d_Target, NUMPAT*NUMOUT*sizeof(float));
	cudaMalloc((void **)&d_Output, NUMPAT*NUMOUT*sizeof(float));
	cudaMalloc((void **)&d_DeltaO, NUMOUT*sizeof(float));
	cudaMalloc((void **)&d_BError, NUMOUT*sizeof(float));
	cudaMalloc((void **)&d_BErrorAcc, NUMOUT*sizeof(float));
	cudaMalloc((void **)&d_WeightHO, NUMOUT*NUMHID*sizeof(float));
	//Copiar Host to Device
	cudaMemcpy(d_Target, Target, NUMPAT*NUMOUT*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_WeightHO, WeightHO, NUMOUT*NUMHID*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_DeltaO, DeltaO, NUMOUT*sizeof(float), cudaMemcpyHostToDevice);

	//FOR 3
	float *d_DeltaH;
	float *d_DeltaWeightIH;
	//Assignar espai en la memoria de la grafica
	cudaMalloc((void **)&d_DeltaH, NUMHID*sizeof(float));
	cudaMalloc((void **)&d_DeltaWeightIH, NUMHID*NUMIN*sizeof(float));
	//cudaMalloc((void **)&d_WeightHO, NUMOUT*NUMHID*sizeof(float));
	//Copiar Host to Device
	cudaMemcpy(d_DeltaWeightIH, DeltaWeightIH, NUMHID*NUMIN*sizeof(float), cudaMemcpyHostToDevice); //posible inicialitzacio en el kernel


	//FOR
	float *d_DeltaWeightHO;
	//Assignar espai en la memoria de la grafica
	cudaMalloc((void **)&d_DeltaWeightHO, NUMOUT*NUMHID*sizeof(float));
	//Copiar Host to Device
	cudaMemcpy(d_DeltaWeightHO, DeltaWeightHO, NUMOUT*NUMHID*sizeof(float), cudaMemcpyHostToDevice);

	//END:Declaració matrius DEVICE

	for( int epoch = 0 ; epoch < 2000 ; epoch++) {    // iterate weight updates
		for( int p = 0 ; p < NUMPAT ; p++ )   // randomize order of individuals
				ranpat[p] = p ;
		for( int p = 0 ; p < NUMPAT ; p++) {
			int x = rando();
			int np = (x*x)%NUMPAT;
			int op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
		}
    float ErrorTotal = 0.0;
		Error = BError = 0.0;
		float BErrorAux[NUMOUT];

		printf("."); fflush(stdout);

		for (int nb = 0; nb < NUMPAT/BSIZE; nb++) { // repeat for all batches

		BError = 0.0;

    for(int x= 0; x < NUMOUT; x++)
		BErrorAux[x]= 0.0;

		cudaMemcpy(d_BErrorAcc, BErrorAux, NUMOUT*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_BError, BErrorAux, NUMOUT*sizeof(float), cudaMemcpyHostToDevice);

		for(int i = 0; i<NUMOUT; i++) BErrorAux[i] = 0;

			for( int np = nb*BSIZE ; np < (nb + 1)*BSIZE ; np++ ) {    // repeat for all the training patterns within the batch
				float DeltaO[NUMOUT], DeltaH[NUMHID];
				int p = ranpat[np];


				//cudaMemcpy(d_tSet, tSet, NUMPAT*NUMIN*sizeof(char), cudaMemcpyHostToDevice);
				Kernel_for_1<<<NUMHID, NUMIN>>>(d_WeightIH, d_tSet, p, d_SumH); // FOR 1
				Kernel_SumH<<<1, NUMHID>>>(d_SumH, d_Hidden, p);
				/*
				//FOR 1
				for( int j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations

					//1 FUNCIO KERNEL
					float SumH = 0.0;
					for( int i = 0 ; i < NUMIN ; i++ ) SumH += tSet[p][i] * WeightIH[j][i]; //TREBALLAR A AQUEST NIVELL AMB CUDA

					//2 FUNCIO KERNEL
					Hidden[p][j] = 1.0/(1.0 + exp(-SumH)) ;
				}
				*/

				//cudaMemcpy(Hidden, d_Hidden, NUMPAT*NUMHID*sizeof(float), cudaMemcpyDeviceToHost);


				//cudaMemcpy(d_BError, BErrorAux, NUMOUT*sizeof(float), cudaMemcpyHostToDevice);

				Kernel_for_2<<<NUMOUT, NUMHID>>>(d_WeightHO, d_Hidden, d_Target, d_Output, d_DeltaO, p, d_BError);

				Kernel_BError<<<1, NUMOUT>>>(d_BError, d_BErrorAcc); //resultat a d_BError[0]
				/*
				//FOR 2
				for( int k = 0. ; k < NUMOUT ; k++ ) {    // compute output unit activations and errors
					float SumO = 0.0;
					for( int j = 0 ; j < NUMHID ; j++ ) SumO += Hidden[p][j] * WeightHO[k][j] ;
					Output[p][k] = 1.0/(1.0 + exp(-SumO)) ;   // Sigmoidal Outputs
					BError += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   // SSE

					//CUDA CALCUL DE NOUS VALORS A TRAVÉS DE MATRIUS
					DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   // Sigmoidal Outputs, SSE
				}
				*/
				//cudaMemcpy(Output, d_Output, NUMPAT*NUMOUT*sizeof(float), cudaMemcpyDeviceToHost);
				//cudaMemcpy(Target, d_Target, NUMPAT*NUMOUT*sizeof(float), cudaMemcpyDeviceToHost);
				//cudaMemcpy(Output, d_Output, NUMPAT*NUMOUT*sizeof(float), cudaMemcpyDeviceToHost);
				//printf("%f \n",BErrorAux[0]);
				//printf("%f \n", BError);

				//cudaMemcpy(DeltaO, d_DeltaO, NUMOUT*sizeof(float), cudaMemcpyDeviceToHost);
				Kernel_for_3<<<NUMHID, NUMOUT>>>(d_Hidden, d_DeltaH, p, d_WeightHO, d_DeltaO);

				/*
				//FOR 3
				for( int j = 0 ; j < NUMHID ; j++ ) {     // update delta weights DeltaWeightIH
					float SumDOW = 0.0 ;
					for( int k = 0 ; k < NUMOUT ; k++ ) SumDOW += WeightHO[k][j] * DeltaO[k]; //KERNEL_3
					DeltaH[j] = SumDOW * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
				}
				*/

				//cudaMemcpy(DeltaH, d_DeltaH, NUMHID*sizeof(float), cudaMemcpyDeviceToHost);
				//cudaMemcpy(DeltaO, d_DeltaO, NUMOUT*sizeof(float), cudaMemcpyDeviceToHost);
				Kernel_for_3_2<<<NUMHID, NUMIN>>>(d_DeltaH, p, d_DeltaWeightIH, eta, alpha, d_tSet);
				//cudaMemcpy(Hidden, d_Hidden, NUMPAT*NUMHID*sizeof(float), cudaMemcpyDeviceToHost);


				//float value = 0.0;
				//for(int i = 0;i<NUMHID;i++) value += DeltaH[i];
				//printf("Value: %f \n", value);
				/*
				//FOR 3-2
				for(int j = 0; j< NUMHID;j++)
					for(int i = 0; i< NUMIN; i++)
						DeltaWeightIH[j][i] = eta * tSet[p][i] * DeltaH[j] + alpha * DeltaWeightIH[j][i];
				*/
				Kernel_for_4<<<NUMOUT, NUMHID>>>(d_Hidden, d_DeltaO, p, d_DeltaWeightHO, eta, alpha);



				/*
				//FOR 4
				for( int k = 0 ; k < NUMOUT ; k ++ )    // update delta weights DeltaWeightHO
					for( int j = 0 ; j < NUMHID ; j++ )
						DeltaWeightHO[k][j] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
				*/
			}
			cudaMemcpy(BErrorAux, d_BError, NUMOUT*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(DeltaWeightHO, d_DeltaWeightHO, NUMOUT*NUMHID*sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(DeltaWeightIH, d_DeltaWeightIH, NUMHID*NUMIN*sizeof(float), cudaMemcpyDeviceToHost);
			Error += BErrorAux[0];
			//BError = BErrorAux[0];
			//Error += BError;
			//cudaMemcpy(d_WeightIH, WeightIH, NUMHID*NUMIN*sizeof(float), cudaMemcpyHostToDevice);
			//cudaMemcpy(d_WeightHO, WeightHO, NUMOUT*NUMHID*sizeof(float), cudaMemcpyHostToDevice);

		//	Kernel_for_WeightIH<<< gridIH, blocksIH>>>(d_WeightIH, d_DeltaWeightIH);
			//Kernel_for_WeightHO<<< gridHO, blocksHO>>>(d_WeightHO, d_DeltaWeightHO);

			Kernel_for_WeightIH<<< NUMHID, NUMIN>>>(d_WeightIH, d_DeltaWeightIH);
		//	Kernel_for_WeightHO<<< NUMOUT, NUMHID>>>(d_WeightHO, d_DeltaWeightHO);


			//DeltaWeightIH a WeightIH
		/*	for( int j = 0 ; j < NUMHID ; j++ )     // update weights WeightIH
				for( int i = 0 ; i < NUMIN ; i++ )
					WeightIH[j][i] += DeltaWeightIH[j][i];*/

			//DeltaWeightHO a WeightHO
			for( int k = 0 ; k < NUMOUT ; k ++ )    // update weights WeightHO
				for( int j = 0 ; j < NUMHID ; j++ )
					WeightHO[k][j] += DeltaWeightHO[k][j] ;


			//cudaMemcpy(d_WeightIH, WeightIH, NUMHID*NUMIN*sizeof(float), cudaMemcpyHostToDevice);
			//cudaMemcpy(d_WeightHO, WeightHO, NUMOUT*NUMHID*sizeof(float), cudaMemcpyHostToDevice);

			//cudaMemcpy(WeightIH, d_WeightIH, NUMHID*NUMIN*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(WeightHO, d_WeightHO, NUMOUT*NUMHID*sizeof(float), cudaMemcpyDeviceToHost);
		}
		Error = Error/((NUMPAT/BSIZE)*BSIZE);	//mean error for the last epoch
		if( !(epoch%100) ) printf("\nEpoch %-5d :   Error = %f \n", epoch, Error) ;
		if( Error < 0.0004 ) {
			printf("\nEpoch %-5d :   Error = %f \n", epoch, Error) ; break ;  // stop learning when 'near enough'
		}
	}
	//freeDeltaWeights(DeltaWeightIH, DeltaWeightHO);
	//freeTSet( NUMPAT, tSet );

	for( int p = 0 ; p < NUMPAT ; p++ ) {
		printf("\n%d\t", p) ;
		for( int k = 0 ; k < NUMOUT ; k++ ) {
				printf("%f\t%f\t", Target[p][k], Output[p][k]) ;
		}
	}
	printf("\n");
}

void printRecognized(int p, float Output[]){
	int imax = 0;

	for( int i = 1; i < NUMOUT; i++)
		if ( Output[i] > Output[imax] ) imax = i;
	printf("El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p]);
	if ( imax == Validation[p] ) total++;
    for( int k = 0 ; k < NUMOUT ; k++ )
        	printf("\t%f\t", Output[k]) ;
    printf("\n");
}
/*
void runN(){
	char **rSet;
	char *fname[NUMRPAT];

	if( (rSet = loadPatternSet(NUMRPAT, "optdigits.cv", 0)) == NULL){
		printf("Error!!\n");
		exit(-1);
	}

	float Hidden[NUMHID], Output[NUMOUT];

    	for( int p = 0 ; p < NUMRPAT ; p++ ) {    // repeat for all the recognition patterns
        	for( int j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
            		float SumH = 0.0;
            		for( int i = 0 ; i < NUMIN ; i++ ) SumH += rSet[p][i] * WeightIH[j][i];
            		Hidden[j] = 1.0/(1.0 + exp(-SumH)) ;
        	}

        	for( int k = 0 ; k < NUMOUT ; k++ ) {    // compute output unit activations
            		float SumO = 0.0;
            		for( int j = 0 ; j < NUMHID ; j++ ) SumO += Hidden[j] * WeightHO[k][j] ;
            		Output[k] = 1.0/(1.0 + exp(-SumO)) ;   // Sigmoidal Outputs
        	}
        	printRecognized(p, Output);
    	}

	printf("\nTotal encerts = %d\n", total);

	freeTSet( NUMRPAT, rSet );
}
*/
int main() {
	clock_t start = clock();
	srand(start); 		//Comentat porta a resultats deterministes (però en el cas real ha d'aparéixer)
	trainN();
	//runN();

	//freeWeights(WeightIH, WeightHO);
	clock_t end = clock();
	printf("\n\nGoodbye! (%f sec)\n\n", (end-start)/(1.0*CLOCKS_PER_SEC)) ;

    	return 1 ;
}

/*******************************************************************************/
