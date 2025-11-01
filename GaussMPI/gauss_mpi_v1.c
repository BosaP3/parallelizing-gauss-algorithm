#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <mpi.h>

void showMatrix(int n, double *A);
void saveFiles(int n, double *A, double *b);
void saveResult(double *A, double *b, double *x, int n);
int  testLinearSystem(double *A, double *b, double *x, int n);
void generateLinearSystem(int n, double *A, double *b);
void loadLinearSystem(int n, double *A, double *b);
void solveLinearSystem(double *A_local, double *b_local, double *x, int n, int meurank, int P);

int main(int argc, char **argv) {
    int meurank, P; 
    int nerros = 0;
    
    double *A_full = NULL, *b_full = NULL, *x_vec = NULL; 
    double *A_local = NULL, *b_local = NULL;              
    int local_rows; 

    double tempo_inicial, tempo_final; 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &meurank);

    int n;

    // Controle de execução
    if (P != 2 && P != 4 && P != 8 && P != 16 && P != 32) {
        if (meurank == 0) {
            fprintf(stderr, "Erro: Programa deve ser executado com 2, 4, 8, 16 ou 32 processos.\n");
        }
        MPI_Finalize(); 
        exit(1);        
    }

    if (meurank == 0) {
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Controle de execução
    if (meurank == 0) {
        printf("Iniciando execucao com N=%d e P=%d processos.\n", n, P);
    }

    if (meurank == 0) {
        A_full = (double *) malloc(n * n * sizeof(double));
        b_full = (double *) malloc(n * sizeof(double));
        x_vec  = (double *) malloc(n * sizeof(double));
        loadLinearSystem(n, A_full, b_full);
    }

    local_rows = n / P;
    if (meurank < (n % P)) {
        local_rows++;
    }

    A_local = (double *) malloc(local_rows * n * sizeof(double));
    b_local = (double *) malloc(local_rows * sizeof(double));

    // Sincroniza e pega o tempo inicial
    MPI_Barrier(MPI_COMM_WORLD);
    if (meurank == 0) {
        tempo_inicial = MPI_Wtime(); 
    }

    solveLinearSystem(A_local, b_local, x_vec, n, meurank, P);

    // Tempo final
    MPI_Barrier(MPI_COMM_WORLD); 
    if (meurank == 0) {
        tempo_final = MPI_Wtime();
        printf("Tempo de execucao: %f segundos\n", tempo_final - tempo_inicial);
    }

    if (meurank == 0) {
        loadLinearSystem(n, A_full, b_full);
        
        nerros += testLinearSystem(A_full, b_full, x_vec, n);
        printf("Errors=%d\n", nerros);
        
        saveResult(A_full, b_full, x_vec, n);

        free(A_full);
        free(b_full);
        free(x_vec);
    }

    free(A_local);
    free(b_local);

    MPI_Finalize();
    
    return EXIT_SUCCESS;
}

void solveLinearSystem(double *A_local, double *b_local, 
                       double *x, int n, int meurank, int P) 
{
    int i, j, k; 
    
    double *A_full_temp = NULL; 
    double *b_full_temp = NULL; 
    MPI_Status status; 

    int rows_per_p = n / P;
    int remainder = n % P;
    
    int my_num_rows = rows_per_p + (meurank < remainder ? 1 : 0);
    
    int my_first_row = (rows_per_p * meurank) + (meurank < remainder ? meurank : remainder);

    if (meurank == 0) {
        A_full_temp = (double *) malloc(n * n * sizeof(double));
        b_full_temp = (double *) malloc(n * sizeof(double));
        loadLinearSystem(n, A_full_temp, b_full_temp); 

        int current_row = 0;
        for (int dest = 0; dest < P; dest++) {
            
            int rows_to_send = rows_per_p + (dest < remainder ? 1 : 0);
            
            if (dest == 0) {
                // copia localmente
                memcpy(A_local, &A_full_temp[current_row * n], rows_to_send * n * sizeof(double));
                memcpy(b_local, &b_full_temp[current_row], rows_to_send * sizeof(double));
            } else {
                // Envia um BLOCO contíguo de 'rows_to_send' linhas
                MPI_Send(&A_full_temp[current_row * n], rows_to_send * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
                MPI_Send(&b_full_temp[current_row], rows_to_send, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            }
            current_row += rows_to_send;
        }
        free(A_full_temp); 
        free(b_full_temp);
    } 
    else {
        // Recebe o bloco de 'my_num_rows' linhas
        MPI_Recv(A_local, my_num_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(b_local, my_num_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }
    
    double *pivot_row = (double *) malloc(n * sizeof(double));
    double pivot_b;
    
    for (i = 0; i < (n - 1); i++) {

        int owner;
        int pivot_threshold = remainder * (rows_per_p + 1);
        
        if (i < pivot_threshold) {
            owner = i / (rows_per_p + 1);
        } else {
            owner = remainder + (i - pivot_threshold) / rows_per_p;
        }

        if (meurank == owner) {
            int local_idx = i - my_first_row; 
            memcpy(pivot_row, &A_local[local_idx * n], n * sizeof(double));
            pivot_b = b_local[local_idx];
        }

        MPI_Bcast(pivot_row, n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        for (int j_local = 0; j_local < my_num_rows; j_local++) {
            
            int j_global = my_first_row + j_local;

            if (j_global > i) {
                double ratio = A_local[j_local * n + i] / pivot_row[i];
                for (k = i; k < n; k++) { 
                    A_local[j_local * n + k] -= (ratio * pivot_row[k]);
                }
                b_local[j_local] -= (ratio * pivot_b);
            }
        }
    }
    
    free(pivot_row);
    
    if (meurank == 0) {
        A_full_temp = (double *) malloc(n * n * sizeof(double));
        b_full_temp = (double *) malloc(n * sizeof(double));

        int current_row = 0;
        for (int source = 0; source < P; source++) {
           
            int rows_to_recv = rows_per_p + (source < remainder ? 1 : 0);
            
            if (source == 0) {
                memcpy(&A_full_temp[current_row * n], A_local, rows_to_recv * n * sizeof(double));
                memcpy(&b_full_temp[current_row], b_local, rows_to_recv * sizeof(double));
            } else {
                MPI_Recv(&A_full_temp[current_row * n], rows_to_recv * n, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&b_full_temp[current_row], rows_to_recv, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
            }
            current_row += rows_to_recv;
        }
    } 
    else {
        MPI_Send(A_local, my_num_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(b_local, my_num_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    if (meurank == 0) {
        x[n - 1] = b_full_temp[n - 1] / A_full_temp[(n - 1) * n + n - 1];
        for (i = (n - 2); i >= 0; i--) {
            double temp = b_full_temp[i];
            for (j = (i + 1); j < n; j++) {
                temp -= (A_full_temp[i * n + j] * x[j]);
            }
            x[i] = temp / A_full_temp[i * n + i];
        }
        
        free(A_full_temp);
        free(b_full_temp);
    }
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