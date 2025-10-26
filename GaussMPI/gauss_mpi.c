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

/* * Esta é a NOVA função paralela. 
 * Note que ela recebe os IDs (meurank, P) e os ponteiros para 
 * as matrizes/vetores LOCAIS de cada processo.
 */
void solveLinearSystem(
    int n, 
    double *A_local, // Matriz local (só as linhas deste processo)
    double *b_local, // Vetor b local (só os B's deste processo)
    double *x,       // Vetor x (calculado e mantido pelo root)
    int meurank,     // ID deste processo
    int P            // Número total de processos
);

int main(int argc, char **argv) {
    int n;
    int meurank, P; // Variáveis para o ID do processo e o número total de processos
    int nerros = 0;
    
    double *A_full = NULL, *b_full = NULL, *x_vec = NULL; // Matrizes completas (só o root usa)
    double *A_local = NULL, *b_local = NULL;              // Matrizes locais (todos usam)
    int local_rows; // Quantidade de linhas que este processo vai gerenciar

    double tempo_inicial, tempo_final; // Variáveis para medição de tempo

    /* 
    * [cite_start]2. MPI_Init: Inicializa o ambiente MPI. [cite: 801, 811]
    * [cite_start]É a primeira função MPI que deve ser chamada. [cite: 811]
    * Ela "acorda" o sistema de comunicação e permite que os
    * processos se reconheçam.
    */
    MPI_Init(&argc, &argv);

    /*
     * [cite_start]3. MPI_Comm_size: Obtém o número total de processos. [cite: 805, 830]
     * Pergunta ao comunicador 'MPI_COMM_WORLD' (que inclui todos os 
     * [cite_start]processos [cite: 737]) quantos processos (P) estão nesta execução.
     */
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    /*
     * [cite_start]4. MPI_Comm_rank: Obtém o ID (rank) deste processo. [cite: 804, 826]
     * Pergunta ao 'MPI_COMM_WORLD' qual é o ID (meurank) deste 
     * [cite_start]processo específico, que vai de 0 até (P-1). [cite: 727]
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &meurank);


    // --- Carregamento e Distribuição de Dados ---
    
    // APENAS o processo 'root' (rank 0) lê o 'n' do teclado
    if (meurank == 0) {
        printf("Digite a ordem da matriz (n): ");
        scanf("%d", &n);
    }

    /*
     * [cite_start]5. MPI_Bcast (Broadcast): Envia um dado de UM para TODOS. [cite: 36, 101]
     * O processo 'root' (rank 0) envia o valor 'n' que ele leu 
     * [cite_start]para todos os outros processos no 'MPI_COMM_WORLD'. [cite: 114, 118]
     * Isso garante que todos os processos saibam o tamanho do problema.
     * [cite_start]Argumentos: (ponteiro_dado, qtd_elementos, tipo_dado, rank_raiz, comunicador) [cite: 115-119]
     */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Alocação de memória
    // O 'root' (0) aloca a matriz completa para carregar os dados
    if (meurank == 0) {
        A_full = (double *) malloc(n * n * sizeof(double));
        b_full = (double *) malloc(n * sizeof(double));
        x_vec  = (double *) malloc(n * sizeof(double));
        //generateLinearSystem(n, A_full, b_full); // Se for gerar
        //saveFiles(n, A_full, b_full);            // Se for salvar
        loadLinearSystem(n, A_full, b_full);       // Carrega do disco
    }

    // Todos os processos calculam quantas linhas locais vão armazenar (distribuição cíclica)
    // Ex: n=10, P=4. 
    // P0 (rank 0) pega 3 linhas (0, 4, 8)
    // P1 (rank 1) pega 3 linhas (1, 5, 9)
    // P2 (rank 2) pega 2 linhas (2, 6)
    // P3 (rank 3) pega 2 linhas (3, 7)
    local_rows = n / P;
    if (meurank < (n % P)) {
        local_rows++;
    }

    // Todos os processos alocam suas matrizes LOCAIS
    A_local = (double *) malloc(local_rows * n * sizeof(double));
    b_local = (double *) malloc(local_rows * sizeof(double));


    /*
     * [cite_start]6. MPI_Barrier (Barreira): Sincroniza todos os processos. [cite: 35, 69]
     * Ninguém passa deste ponto até que TODOS os processos tenham 
     * [cite_start]chegado aqui. [cite: 70]
     * Usado para garantir que a medição de tempo comece 
     * (o mais próximo) simultaneamente após as alocações.
     */
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Início da Medição de Tempo ---
    if (meurank == 0) {
        tempo_inicial = MPI_Wtime(); // Pega o tempo de relógio (wall clock)
    }

    // Chama a função de solução paralela
    solveLinearSystem(n, A_local, b_local, x_vec, meurank, P);

    // --- Fim da Medição de Tempo ---
    MPI_Barrier(MPI_COMM_WORLD); // Garante que o 'root' só pare o tempo
                                 // depois que o processo mais lento terminar.
    if (meurank == 0) {
        tempo_final = MPI_Wtime();
        printf("Tempo de execucao: %f segundos\n", tempo_final - tempo_inicial);
    }

    // --- Verificação e Finalização (só o root) ---
    if (meurank == 0) {
        // O A_full foi modificado pela função 'solve'
        // Mas a verificação de erros precisa do A_full *original*.
        // Vamos recarregar o original para testar.
        
        loadLinearSystem(n, A_full, b_full); // Recarrega o A original
        
        nerros += testLinearSystem(A_full, b_full, x_vec, n);
        printf("Errors=%d\n", nerros);
        
        saveResult(A_full, b_full, x_vec, n);

        // Libera memória do root 
        free(A_full);
        free(b_full);
        free(x_vec);
    }

    // Libera memória local de todos os processos
    free(A_local);
    free(b_local);

    /*
     * [cite_start]7. MPI_Finalize: Termina o ambiente MPI. [cite: 803, 819]
     * [cite_start]É a última função MPI a ser chamada. [cite: 820]
     * Desativa o sistema de comunicação.
     */
    MPI_Finalize();
    
    return EXIT_SUCCESS;
}


/*
 * Esta é a função que implementa a Eliminação Gaussiana Paralela.
 * Ela substitui a 'solveLinearSystem' sequencial.
 */
void solveLinearSystem(int n, double *A_local, double *b_local, double *x, int meurank, int P) {
    int i, j, k; // i = pivô global, j = linha local, k = coluna
    int local_row_idx = 0; // Índice para preencher A_local
    int global_row_owner;
    int local_pivot_idx;

    // Buffers para MPI_Send/Recv
    double *A_full_temp = NULL; // Só o root usa
    double *b_full_temp = NULL; // Só o root usa
    
    // Status do MPI_Recv (para obter info da mensagem)
    MPI_Status status; 

    // --- 1. FASE DE DISTRIBUIÇÃO (Scatter Cíclico) ---
    // O 'root' (0) envia as linhas para os processos corretos
    if (meurank == 0) {
        A_full_temp = (double *) malloc(n * n * sizeof(double));
        b_full_temp = (double *) malloc(n * sizeof(double));
        loadLinearSystem(n, A_full_temp, b_full_temp); // Carrega dados p/ distribuir

        local_row_idx = 0;
        for (i = 0; i < n; i++) {
            int dest_rank = i % P; // Estratégia Cíclica
            
            if (dest_rank == 0) {
                // Se o destino sou eu (root), apenas copio localmente
                memcpy(&A_local[local_row_idx * n], &A_full_temp[i * n], n * sizeof(double));
                b_local[local_row_idx] = b_full_temp[i];
                local_row_idx++;
            } else {
                /*
                 * [cite_start]8. MPI_Send: Envia uma mensagem (bloqueante). [cite: 806, 860]
                 * O processo 'root' envia a linha 'i' da matriz e o 
                 * elemento 'i' do vetor 'b' para o processo 'dest_rank'.
                 * [cite_start]O 'tag' (assunto) é 'i' para identificar a linha. [cite: 897]
                 * [cite_start]Argumentos: (ponteiro_dado, qtd, tipo, rank_destino, tag, comunicador) [cite: 864]
                 */
                MPI_Send(&A_full_temp[i * n], n, MPI_DOUBLE, dest_rank, i, MPI_COMM_WORLD);
                MPI_Send(&b_full_temp[i], 1, MPI_DOUBLE, dest_rank, i, MPI_COMM_WORLD);
            }
        }
        free(A_full_temp); // Libera as matrizes temporárias
        free(b_full_temp);
    } 
    // Os outros processos (escravos) recebem suas linhas
    else {
        int my_rows = n / P; // Calcula quantas linhas vai receber
        if (meurank < (n % P)) {
            my_rows++;
        }

        for (j = 0; j < my_rows; j++) {
            // Calcula qual linha global 'i' ele deve esperar
            int global_row_i = (j * P) + meurank; 
            
            /*
             * [cite_start]9. MPI_Recv: Recebe uma mensagem (bloqueante). [cite: 807, 902]
             * O processo 'escravo' espera (bloqueado) até receber 
             * [cite_start]uma mensagem do 'root' (rank 0) com o 'tag' 'global_row_i'. [cite: 904]
             * Ele salva os dados recebidos em seu 'A_local' e 'b_local'.
             * [cite_start]MPI_ANY_TAG poderia ser usado se a ordem não importasse. [cite: 1054]
             * [cite_start]Argumentos: (buffer_rec, qtd, tipo, rank_origem, tag, comm, &status) [cite: 907]
             */
            MPI_Recv(&A_local[j * n], n, MPI_DOUBLE, 0, global_row_i, MPI_COMM_WORLD, &status);
            MPI_Recv(&b_local[j], 1, MPI_DOUBLE, 0, global_row_i, MPI_COMM_WORLD, &status);
        }
    }

    // --- 2. FASE DE ELIMINAÇÃO GAUSSIANA (Computação Paralela) ---
    
    // Aloca um buffer em TODOS os processos para receber a linha-pivô
    double *pivot_row = (double *) malloc(n * sizeof(double));
    double pivot_b;

    // O loop 'i' (pivô) é sequencial e executado por todos
    for (i = 0; i < (n - 1); i++) {
        
        // 1. Descobrir quem é o "dono" da linha pivô 'i'
        global_row_owner = i % P;

        // 2. O processo "dono" copia sua linha pivô local para o buffer
        if (meurank == global_row_owner) {
            local_pivot_idx = i / P; // Calcula o índice local da linha pivô
            memcpy(pivot_row, &A_local[local_pivot_idx * n], n * sizeof(double));
            pivot_b = b_local[local_pivot_idx];
        }

        /*
         * 10. MPI_Bcast (Novamente): O 'dono' envia (broadcast) a 
         * [cite_start]linha pivô para TODOS os outros processos. [cite: 100, 101]
         * Isso garante que todos os processos tenham a linha 'i' 
         * necessária para calcular o 'ratio' e atualizar suas próprias linhas.
         */
        MPI_Bcast(pivot_row, n, MPI_DOUBLE, global_row_owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, global_row_owner, MPI_COMM_WORLD);

        // 3. Computação Paralela (Cada processo atualiza SUAS linhas)
        int my_rows = n / P;
        if (meurank < (n % P)) {
            my_rows++;
        }

        for (j = 0; j < my_rows; j++) {
            // Descobre o índice global da linha local 'j'
            int global_row_j = (j * P) + meurank;

            // Só atualiza linhas que estão ABAIXO do pivô 'i'
            if (global_row_j > i) {
                double ratio = A_local[j * n + i] / pivot_row[i];
                for (k = i; k < n; k++) { // O loop 'k' é o mesmo da versão sequencial
                    A_local[j * n + k] -= (ratio * pivot_row[k]);
                }
                b_local[j] -= (ratio * pivot_b);
            }
        }
        
        // Tecnicamente, uma barreira aqui (MPI_Barrier) não é necessária
        // porque o MPI_Bcast no início do próximo loop 'i' já 
        // força a sincronização.
    }
    
    free(pivot_row); // Libera o buffer do pivô

    // --- 3. FASE DE GATHER (Coleta de Dados para Retro-substituição) ---
    // A retro-substituição (O(N^2)) é rápida e faremos sequencialmente no 'root'.
    // Precisamos enviar os dados (A_local e b_local modificados) de volta para o 'root'.
    
    if (meurank == 0) {
        A_full_temp = (double *) malloc(n * n * sizeof(double));
        b_full_temp = (double *) malloc(n * sizeof(double));

        local_row_idx = 0;
        for (i = 0; i < n; i++) {
            int source_rank = i % P; // De quem esperamos receber a linha 'i'?
            
            if (source_rank == 0) {
                // Se sou eu (root), apenas copio localmente
                memcpy(&A_full_temp[i * n], &A_local[local_row_idx * n], n * sizeof(double));
                b_full_temp[i] = b_local[local_row_idx];
                local_row_idx++;
            } else {
                // Se for de outro, eu recebo (MPI_Recv)
                MPI_Recv(&A_full_temp[i * n], n, MPI_DOUBLE, source_rank, i, MPI_COMM_WORLD, &status);
                MPI_Recv(&b_full_temp[i], 1, MPI_DOUBLE, source_rank, i, MPI_COMM_WORLD, &status);
            }
        }
    } 
    // Os 'escravos' enviam suas linhas locais de volta para o 'root'
    else {
        int my_rows = n / P;
        if (meurank < (n % P)) {
            my_rows++;
        }

        for (j = 0; j < my_rows; j++) {
            int global_row_i = (j * P) + meurank; // Qual linha global eu estou enviando
            // Envia (MPI_Send) a linha 'j' local (que é a 'global_row_i' global)
            MPI_Send(&A_local[j * n], n, MPI_DOUBLE, 0, global_row_i, MPI_COMM_WORLD);
            MPI_Send(&b_local[j], 1, MPI_DOUBLE, 0, global_row_i, MPI_COMM_WORLD);
        }
    }

    // --- 4. FASE DE RETRO-SUBSTITUIÇÃO (Sequencial no Root) ---
    if (meurank == 0) {
        /* * O 'root' (0) agora tem a matriz 'A_full_temp' triangularizada
         * e o 'b_full_temp' modificado. 
         * Podemos usar o código EXATAMENTE sequencial para resolver 'x'.
         */
        
        // Copiado do seu 'gauss_sequencial.c'
        x[n - 1] = b_full_temp[n - 1] / A_full_temp[(n - 1) * n + n - 1];
        for (i = (n - 2); i >= 0; i--) {
            double temp = b_full_temp[i];
            for (j = (i + 1); j < n; j++) {
                temp -= (A_full_temp[i * n + j] * x[j]);
            }
            x[i] = temp / A_full_temp[i * n + i];
        }
        
        // Libera os temporários finais
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