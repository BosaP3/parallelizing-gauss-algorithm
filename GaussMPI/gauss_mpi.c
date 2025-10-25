#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <mpi.h> // <--- ADICIONAR ESTA LINHA

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* Protótipos */
void solveLinearSystem(const double *A, const double *b, double *x, int n);
void loadLinearSystem(int n, double *A, double *b);
int testLinearSystem(double *A, double *b, double *x, int n);
void saveResult(double *A, double *b, double *x, int n);
/* Se você tiver generateLinearSystem/saveFiles, declare também:
   void generateLinearSystem(int n, double *A, double *b);
   void saveFiles(int n, double *A, double *b);
*/

int main(int argc, char **argv) {
    int n;
    int nerros = 0;
    int rank, size;
    double t1, t2; /* Para medição de tempo */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Rank 0 lê o tamanho 'n' do stdin */
    if (rank == 0) {
        if (scanf("%d", &n) != 1) {
            fprintf(stderr, "Erro ao ler 'n' do stdin\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    /* Rank 0 transmite 'n' para todos os outros processos */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Todos os processos alocam memória */
    double *A = (double *) malloc(n * n * sizeof(double));
    double *b = (double *) malloc(n * sizeof(double));
    double *x = (double *) malloc(n * sizeof(double));
    if (A == NULL || b == NULL || x == NULL) {
        fprintf(stderr, "Erro de alocação de memória\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Apenas Rank 0 carrega os dados */
    if (rank == 0) {
        /* generateLinearSystem(n, &A[0], &b[0]); */
        /* saveFiles(n, &A[0], &b[0]); */
        loadLinearSystem(n, &A[0], &b[0]);
    }

    /* Sincroniza todos os processos antes de iniciar o timer */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Rank 0 inicia o timer (medindo o tempo da função de solução) */
    if (rank == 0) {
        t1 = MPI_Wtime();
    }

    /* Todos os processos chamam a função de solução paralela */
    solveLinearSystem(&A[0], &b[0], &x[0], n);

    /* Rank 0 para o timer */
    if (rank == 0) {
        t2 = MPI_Wtime();

        printf("--------------------------------------------------\n");
        printf("Tempo de execução (solveLinearSystem): %f segundos\n", t2 - t1);
        printf("--------------------------------------------------\n");

        /* Apenas Rank 0 testa e salva os resultados, pois 'x' só existe nele */
        nerros += testLinearSystem(&A[0], &b[0], &x[0], n);
        printf("Errors=%d\n", nerros);
        saveResult(&A[0], &b[0], &x[0], n);
    }

    /* Libera memória em todos os processos */
    free(A);
    free(b);
    free(x);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void solveLinearSystem(const double *A, const double *b, double *x, int n) {
    int rank, size;
    int i, j, count;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Todos os processos alocam suas cópias locais */
    double *Acpy = (double *) malloc(n * n * sizeof(double));
    double *bcpy = (double *) malloc(n * sizeof(double));
    if (Acpy == NULL || bcpy == NULL) {
        fprintf(stderr, "Erro de alocação em solveLinearSystem\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Rank 0 copia os dados de entrada para seus buffers */
    if (rank == 0) {
        memcpy(Acpy, A, n * n * sizeof(double));
        memcpy(bcpy, b, n * sizeof(double));
    }

    /* Rank 0 transmite a matriz e o vetor inicial para todos os processos */
    MPI_Bcast(Acpy, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(bcpy, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Gaussian Elimination (Paralela com distribuição cíclica) */
    for (i = 0; i < (n - 1); i++) {

        /* 1. O processo "dono" da linha pivô (i) faz o broadcast dela */
        int pivot_owner = i % size;
        MPI_Bcast(&Acpy[i * n], n, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
        MPI_Bcast(&bcpy[i], 1, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

        /* 2. Todos os processos atualizam suas próprias linhas
           (distribuição cíclica: j = rank, rank + size, rank + 2*size, ...) */
        for (j = rank; j < n; j += size) {
            if (j > i) { /* Apenas linhas abaixo do pivô */
                double ratio = Acpy[j * n + i] / Acpy[i * n + i];
                for (count = i; count < n; count++) {
                    Acpy[j * n + count] -= (ratio * Acpy[i * n + count]);
                }
                bcpy[j] -= (ratio * bcpy[i]);
            }
        }
        /* Sincronização é garantida pelo próximo MPI_Bcast do pivô */
    }

    /* Gather e Back-substitution (Executado apenas no Rank 0) */
    if (rank != 0) {
        for (j = rank; j < n; j += size) {
            /* Envia a linha 'j' da matriz e o elemento 'j' do vetor */
            MPI_Send(&Acpy[j * n], n, MPI_DOUBLE, 0, j, MPI_COMM_WORLD);
            MPI_Send(&bcpy[j], 1, MPI_DOUBLE, 0, j + n, MPI_COMM_WORLD);
        }
    } else {
        MPI_Status status;
        for (int p_source = 1; p_source < size; p_source++) {
            for (j = p_source; j < n; j += size) {
                MPI_Recv(&Acpy[j * n], n, MPI_DOUBLE, p_source, j, MPI_COMM_WORLD, &status);
                MPI_Recv(&bcpy[j], 1, MPI_DOUBLE, p_source, j + n, MPI_COMM_WORLD, &status);
            }
        }

        /* Back-substitution (Sequencial, apenas no Rank 0) */
        x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
        for (i = (n - 2); i >= 0; i--) {
            double temp = bcpy[i];
            for (j = (i + 1); j < n; j++) {
                temp -= (Acpy[i * n + j] * x[j]);
            }
            x[i] = temp / Acpy[i * n + i];
        }
    }

    /* Libera as cópias locais */
    free(Acpy);
    free(bcpy);
}


void showMatrix(int n, double *A){
    int i, j;
    
    for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
		    printf("%.6f\t", A[i * n + j]);
		printf("\n");
	}
}

void saveFiles(int n, double *A, double *b) {
	int i, j;	
    FILE *mat, *vet;
		
    mat = fopen("matrix.in", "w");
	if (mat == NULL){
	    printf("File matrix.in does not open\n");
	    exit(1);
	}
	
	vet = fopen("vector.in", "w");
	if (vet == NULL){
	    printf("File vector.in does not open\n");
	    exit(1);
	}	
			
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
		    fprintf(mat, "%.6f\t", A[i * n + j]);
		fprintf(mat, "\n");

	    fprintf(vet, "%.6f\n", b[i]);
	}
	
	fclose ( mat );
	fclose ( vet );
}

void saveResult(double *A, double *b, double *x, int n) {	
	int i;
    FILE *res;	
    
    res = fopen("result.out", "w");
	if (res == NULL){
	    printf("File result.out does not open\n");
	    exit(1);
	}
	
	for(i=0; i < n; i++){
	    fprintf(res, "%.6f\n", x[i]);
    }
	
	fclose( res );
}

int testLinearSystem(double *A, double *b, double *x, int n) {
	int i, j, c =0;
	double sum = 0;

	for (i = 0; i < n; i++) {
		sum=0;
		for (j = 0; j < n; j++)
			sum += A[i * n + j] * x[j];		
		if (abs(sum - b[i]) >= 0.001) {
		    printf("%f\n", (sum - b[i]) );
			c++;
		}
	}
	return c;
}

void generateLinearSystem(int n, double *A, double *b) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			A[i * n + j] = (1.0 * n + (rand() % n)) / (i + j + 1);
		A[i * n + i] = (10.0 * n) / (i + i + 1);
	}
    
	for (i = 0; i < n; i++)
		b[i] = 1.;		
}

void loadLinearSystem(int n, double *A, double *b) {
	int i, j;	
	FILE *mat, *vet;
		
        mat = fopen("matrix.in", "r");
	if (mat == NULL){
	    printf("File matrix.in does not open\n");
	    exit(1);
	}
	
	vet = fopen("vector.in", "r");
	if (vet == NULL){
	    printf("File vector.in does not open\n");
	    exit(1);
	}	

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			fscanf(mat, "%lf", &A[i * n + j]);
	}

	for (i = 0; i < n; i++)
		fscanf(vet, "%lf", &b[i]);
		
	fclose ( mat );
	fclose ( vet );			
}
