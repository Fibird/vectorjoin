/* $begin mountainmain */
#include <stdlib.h>
#include <stdio.h>
#include "Rdtsc.h"              /* startTimer, stopTimer */
#include <sys/time.h>           /* gettimeofday */
#include <pthread.h>
#include <errno.h>
#include <cuda_runtime.h> 

#define L1 (1<<15)    /* Working set size for L1 cache 32KB */
#define L2 (1<<18)    /* Working set size for L2 cache 256KB */
#define L3 (1<<20)*2.5    /* Working set size for L3 cache 2.5MB */
#define LLC (1<<20)*55    /* Working set size for LLC cache 55MB */
#define MAXELEMS 600000000 
#define random(x) (rand()%x)
#define nthread 40
#define nstream 4

#define THREAD_NUM 1024 //4096
#define BLOCK_NUM 13

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

typedef struct {
    int *data;
    int *vector;
    int data_len;
    int vec_len;
    int result;
} args_t;

void init_data(int *data, int n, int cardinality);
void test(int *data, int *vector, int n, int nthreads);
//long run(int *data, int *vector, int n);
double run(int *data, int *vector, int n);
//long run_on_gpu(int *data, int n, int *vector, int vec_len, bool hasTransferTime);
double run_on_gpu(int *data, int n, int *vector, int vec_len, bool hasTransferTime);

/* $begin mountainmain */
int main(int argc, char **argv)
{
    int * data=(int *)malloc(sizeof(int) * MAXELEMS);      /* foreign key column*/
    printf("vector join(cycles per tuple)\n");
    int step;
    printf("===========================CPU Time Metrix===========================\n");
    for (step = 10; step <= 100; step+=10)
	printf("%d%\t", step);
    printf("\n");
	/*vector join for L1 cache sets*/
	printf("L1\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L1/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run(data,vector,MAXELEMS));
		printf("%4.2lf\t", run(data,vector,MAXELEMS));
	}
	printf("\n");
	/*vector join for L2 cache sets*/
	printf("L2\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L2/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run(data,vector,MAXELEMS));
		printf("%4.2lf\t", run(data,vector,MAXELEMS));
	}
	printf("\n");
	/*vector join for L3 cache sets*/
	printf("L3\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L3/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run(data,vector,MAXELEMS));
		printf("%4.2lf\t", run(data,vector,MAXELEMS));
	}
	printf("\n");
	/*vector join for LLC cache sets*/
	printf("LLC\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=LLC/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run(data,vector,MAXELEMS));
		printf("%4.2lf\t", run(data,vector,MAXELEMS));
	}
    printf("\n");
    printf("===========================GPU Time Metrix(has Transfer Time)===========================\n");
    for (step = 10; step <= 100; step+=10)
	printf("%d%\t", step);
    printf("\n");
	/*vector join for L1 cache sets*/
	printf("L1\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L1/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run_on_gpu(data, MAXELEMS, vector, vector_len, true));
		printf("%4.2lf\t", run_on_gpu(data, MAXELEMS, vector, vector_len, true));
	}
	printf("\n");
	/*vector join for L2 cache sets*/
	printf("L2\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L2/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run_on_gpu(data, MAXELEMS, vector, vector_len, true));
		printf("%4.2lf\t", run_on_gpu(data, MAXELEMS, vector, vector_len, true));
	}
	printf("\n");
	/*vector join for L3 cache sets*/
	printf("L3\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L3/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run_on_gpu(data, MAXELEMS, vector, vector_len, true));
		printf("%4.2lf\t", run_on_gpu(data, MAXELEMS, vector, vector_len, true));
	}
	printf("\n");
	/*vector join for LLC cache sets*/
	printf("LLC\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=LLC/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run_on_gpu(data, MAXELEMS, vector, vector_len, true));
		printf("%4.2lf\t", run_on_gpu(data, MAXELEMS, vector, vector_len, true));
	}
	printf("\n");
    printf("===========================GPU Time Metrix(No Transfer Time)===========================\n");
    for (step = 10; step <= 100; step+=10)
	printf("%d%\t", step);
    printf("\n");
	/*vector join for L1 cache sets*/
	printf("L1\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L1/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run_on_gpu(data, MAXELEMS, vector, vector_len, false));
		printf("%4.2lf\t", run_on_gpu(data, MAXELEMS, vector, vector_len, false));
	}
	printf("\n");
	/*vector join for L2 cache sets*/
	printf("L2\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L2/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run_on_gpu(data, MAXELEMS, vector, vector_len, false));
		printf("%4.2lf\t", run_on_gpu(data, MAXELEMS, vector, vector_len, false));
	}
	printf("\n");
	/*vector join for L3 cache sets*/
	printf("L3\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=L3/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run_on_gpu(data, MAXELEMS, vector, vector_len, false));
		printf("%4.2lf\t", run_on_gpu(data, MAXELEMS, vector, vector_len, false));
	}
	printf("\n");
	/*vector join for LLC cache sets*/
	printf("LLC\t");
    for (step = 10; step <= 100; step+=10){
		int i,vector_len=LLC/4*step/100;
		init_data(data, MAXELEMS,vector_len); /* Initialize foreign key elements in data */
		int * vector=(int *)malloc(sizeof(int) * vector_len);   /* Initialize primary key vector elements in vector */
		for (i=0;i<vector_len;i++)
			vector[i]=1;
		//printf("%ld\t", run_on_gpu(data, MAXELEMS, vector, vector_len, false));
		printf("%4.2lf\t", run_on_gpu(data, MAXELEMS, vector, vector_len, false));
	}
	printf("\n");
    exit(0);
}
/* init_data - initializes the array */
void init_data(int *data, int n, int cardinality)
{
    int i;
    for (i = 0; i < n; i++)
		data[i] = random(cardinality);
}

void *vector_thread(void *args) /* The test function */
{
    args_t *param = (args_t*) args;
    int *data = param->data;
    int *vector = param->vector;
    int n = param->data_len;

    int i;
    int result = 0; 
    volatile int sink=0; 
    for (i = 0; i < n; i++) {
		if(vector[data[i]]==1)  /*foreign key referenced vector position*/
			result++;
    }
    sink = result; /* So compiler doesn't optimize away the loop */
    param->result = sink;
}

void test(int *data, int *vector, int n, int nthreads) {
    int numDatathr = n / nthreads;
    int result = 0;
    args_t args[nthreads];
    int numData = n;
    pthread_t tid[nthreads];
    int i; 
    for (i = 0; i < nthreads; i++) {
        args[i].data = data + i * numDatathr;
        args[i].vector = vector;
        args[i].data_len = (i == (nthreads - 1) ? numData : numDatathr);
        args[i].result = 0;
        numData -= numDatathr;
        int rv = pthread_create(&tid[i], NULL, vector_thread, (void*)&args[i]);
        if (rv) {
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
    }

    for (i = 0; i < nthreads; i++) {
        pthread_join(tid[i], NULL);
        /* sum up results */
        result += args[i].result;
    }
    //printf("[%ld]",result);
}

//long run(int *data, int *vector, int n)
double run(int *data, int *vector, int n)
{   
    //long cycles_per_tuples;
    double cycles_per_tuples;
    uint64_t timer;
    test(data, vector,n, nthread);                     /* warm up the cache */       //line:mem:warmup
    startTimer(&timer);
    test(data, vector,n, nthread);                     /* test for cache locality */    
    stopTimer(&timer); 
    // cycles per tuples
    //cycles_per_tuples = timer / n;  
    cycles_per_tuples = (double) timer / (double) n;  
    return cycles_per_tuples; 
}
/* $end mountainfuns */


// Data for computing stored in global memory
// Result stored in shared memory
__global__ void cuda_vectorjoin_thread(int *data, int *vector, int n, int *CountResult) {
    __shared__ int shared[THREAD_NUM];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;
    shared[tid] = 0;

    for (i = bid * THREAD_NUM + tid; i < n; i += BLOCK_NUM * THREAD_NUM) {
        if (vector[data[i]] == 1) {
            shared[tid] += 1;
        }
    }

    __syncthreads();

    if (tid == 0) {
        for (i = 1; i < THREAD_NUM; i++) {
            shared[0] += shared[i];
        }
        CountResult[bid] = shared[0];
    }
}

//long run_on_gpu(int *data, int n, int *vector, int vec_len, bool hasTransferTime) {
double run_on_gpu(int *data, int n, int *vector, int vec_len, bool hasTransferTime) {
    int result = 0;
    int *dev_data, *dev_vector;
    int *dev_results[nstream], *results;
    uint64_t timer1, timer2;
    //long cycles_per_tuples1, cycles_per_tuples2;
    double cycles_per_tuples1, cycles_per_tuples2;
    int numDatastr = n / nstream;
    int numData = n;
    cudaStream_t streams[nstream];
    int i, j;

    // Malloc host memory for result set
    results = (int*)malloc(sizeof(int) * BLOCK_NUM);

    // Malloc device memory for data and vector
    checkCuda(cudaMalloc((void**)&dev_data, sizeof(int) * n));
    checkCuda(cudaMalloc((void**)&dev_vector, sizeof(int) * vec_len));
    
    // Malloc device memory for devcie result set array
    for (i = 0; i < nstream; i++) {
        checkCuda(cudaMalloc((void**)&dev_results[i], sizeof(int) * BLOCK_NUM));
    }

    // Copy host data to device
    checkCuda(cudaMemcpy(dev_data, data, sizeof(int) * n, cudaMemcpyHostToDevice)); 

    // Time for vector transfering and GPU computing
    startTimer(&timer1);        
    // Copy host vector to device
    checkCuda(cudaMemcpy(dev_vector, vector, sizeof(int) * vec_len,  cudaMemcpyHostToDevice)); 
    
    int tmpDataLen = 0;
    startTimer(&timer2);
    for (i = 0; i < nstream; i++) {
        tmpDataLen = (i == (nstream - 1) ? numData : numDatastr);
        numData -= numDatastr;
        // Create cuda stream
        checkCuda(cudaStreamCreate(&streams[i]));
        // Launch kernel function asynchronously
        cuda_vectorjoin_thread<<<BLOCK_NUM, THREAD_NUM, 0, streams[i]>>>(dev_data + i * numDatastr, dev_vector, tmpDataLen, dev_results[i]);
    }

    for (i = 0; i < nstream; i++) {
        // Sync kernels on cuda streams
        checkCuda(cudaStreamSynchronize(streams[i]));
    }
    stopTimer(&timer2); 
    stopTimer(&timer1); 
    
    //cycles_per_tuples1 = timer /  n;
    cycles_per_tuples1 = (double) timer1 / (double) n;
    //cycles_per_tuples2 = timer / n;
    cycles_per_tuples2 = (double) timer2 / (double) n;
     
    for (i = 0; i < nstream; i++) {
        // Copy result set from device to host
        checkCuda(cudaMemcpy(results, dev_results[i], sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost));
        for (j = 0; j < BLOCK_NUM; j++) {
            result += results[j];
        }
    }

    for (i = 0; i < nstream; i++) {
        // free cuda streams
        checkCuda(cudaStreamDestroy(streams[i]));
    }

    // free device memory
    for (i = 0; i < nstream; i++) 
        checkCuda(cudaFree(dev_results[i]));
    checkCuda(cudaFree(dev_vector));
    checkCuda(cudaFree(dev_data));
    //printf("[%d]",result);
    return hasTransferTime ? cycles_per_tuples1 : cycles_per_tuples2;
}

// use default single stream
/*void run_on_gpu(int *data, int n, int *vector, int vec_len) {
    int result = 0;
    int *dev_data, *dev_vector;
    int *dev_results, *results;

    results = (int*)malloc(sizeof(int) * BLOCK_NUM);
    checkCuda(cudaMalloc((void**)&dev_data, sizeof(int) * n));
    checkCuda(cudaMalloc((void**)&dev_vector, sizeof(int) * vec_len));
    checkCuda(cudaMalloc((void**)&dev_results, sizeof(int) * BLOCK_NUM));

    checkCuda(cudaMemcpy(dev_data, data, sizeof(int) * n, cudaMemcpyHostToDevice)); 
    checkCuda(cudaMemcpy(dev_vector, vector, sizeof(int) * vec_len,  cudaMemcpyHostToDevice)); 

    cuda_vectorjoin_thread<<<BLOCK_NUM, THREAD_NUM>>>(dev_data, dev_vector, n, dev_results);

    checkCuda(cudaMemcpy(results, dev_results, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost));
    int i; 
    for (i = 0; i < BLOCK_NUM; i++) {
        result += results[i];
    }
    printf("[%d]\n",result);
}*/
