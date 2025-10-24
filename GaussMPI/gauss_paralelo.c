/*
 * gauss_paralelo.c
 * * Solucao paralela para o sistema de equacoes lineares Ax=b usando
 * Eliminacao Gaussiana com MPI.
 * * Estrategia:
 * 1. Padrao: Fases Paralelas (Distribuicao, Eliminacao, Coleta)
 * 2. Distribuicao de Dados: Cíclica (interleaved) por linhas para balanceamento de carga.
 * - Processo 0 (rank 0) fica com as linhas 0, N, 2N, ...
 * - Processo 1 (rank 1) fica com as linhas 1, N+1, 2N+1, ...
 * 3. Eliminacao:
 * - A cada iteracao 'i', o processo "dono" da linha pivo 'i' (rank i % total_de_processos)
 * envia essa linha para todos os outros via MPI_Bcast.
 * - Cada processo, entao, atualiza suas proprias linhas localmente.
 * 4. Retro-substituicao:
 * - Os dados sao coletados de volta no processo 0 (Gather manual).
 * - O processo 0 resolve a retro-substituicao sequencialmente,
 * pois o custo de paralelizar O(n^2) nao compensa a comunicacao.
 */

 #include <math.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <sys/time.h>
 #include <assert.h>
 #include <mpi.h> // Header do MPI
 
 /* Funcoes sequenciais originais (sem modificacao) */
 void showMatrix(int n, double *A);
 void saveFiles(int n, double *A, double *b);
 void saveResult(double *A, double *b, double *x, int n);
 int  testLinearSystem(double *A, double *b, double *x, int n);
 void generateLinearSystem(int n, double *A, double *b);
 void loadLinearSystem(int n, double *A, double *b);
 
 /* Funcao de resolucao modificada para MPI */
 void solveLinearSystem(const double *A, const double *b, double *x, int n, int meu__rank, int total_de_processos);
 
 
 int main(int argc, char **argv) {
     int n;
     int nerros = 0;
     int meu__rank, total_de_processos;
 
     /* --- INICIALIZACAO MPI --- */
     MPI_Init(&argc, &argv);
     [cite_start]MPI_Comm_rank(MPI_COMM_WORLD, &meu__rank); [cite: 301, 303, 306]
     [cite_start]MPI_Comm_size(MPI_COMM_WORLD, &total_de_processos); [cite: 302, 308, 310]
 
     /* * Verificacao de processos conforme especificacao do trabalho
      * Apenas um processo (rank 0) deve imprimir a mensagem de erro.
      */
     if (total_de_processos != 2 && total_de_processos != 4 && total_de_processos != 8 && total_de_processos != 16 && total_de_processos != 32) {
         if (meu__rank == 0) {
             printf("Erro: Este programa deve ser executado com 2, 4, 8, 16 ou 32 processos.\n");
         }
         [cite_start]MPI_Finalize(); [cite: 287, 289, 291]
         return EXIT_FAILURE;
     }
 
     /* Processo 0 le o 'n' e envia para todos os outros */
     if (meu__rank == 0) {
         scanf("%d", &n);
     }
     [cite_start]MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); [cite: 813, 817-818, 827-832] // Todos recebem 'n'
 
     /* Todos os processos alocam memoria para as matrizes/vetores */
     double *A = (double *) malloc(n * n * sizeof(double));
     double *b = (double *) malloc(n * sizeof(double));
     double *x = (double *) malloc(n * sizeof(double));
 
     /* Apenas o processo 0 le os dados de entrada */
     if (meu__rank == 0) {
         loadLinearSystem(n, &A[0], &b[0]);
     }
 
     /* * Chamada da funcao paralela.
      * Todos os processos devem chamar esta funcao.
      */
     solveLinearSystem(&A[0], &b[0], &x[0], n, meu__rank, total_de_processos);
 
     /* * Apenas o processo 0 (que agora tem a solucao 'x')
      * deve testar e salvar os resultados.
      */
     if (meu__rank == 0) {
         printf("Testando solucao...\n");
         nerros += testLinearSystem(&A[0], &b[0], &x[0], n);
         printf("Errors=%d\n", nerros);
         saveResult(&A[0], &b[0], &x[0], n);
     }
 
     /* --- FINALIZACAO MPI --- */
     free(A);
     free(b);
     free(x);
     [cite_start]MPI_Finalize(); [cite: 287, 289, 291]
     return EXIT_SUCCESS;
 }
 
 /**
  * Funcao principal de resolucao do sistema.
  * Implementa o padrao Fases Paralelas com distribuicao ciclica.
  */
 void solveLinearSystem(const double *A, const double *b, double *x, int n, int meu__rank, int total_de_processos) {
 
     /* Variavel para guardar o status de operacoes de recebimento MPI */
     [cite_start]MPI_Status status_recebimento; [cite: 413-414]
    
     /* Variaveis para medicao de tempo */
     double tempo_inicio, tempo_fim;
 
     /* Todos os processos alocam suas copias locais */
     double *Acpy = (double *) malloc(n * n * sizeof(double));
     double *bcpy = (double *) malloc(n * sizeof(double));
 
     /* Sincroniza todos antes de marcar o tempo (garante que todos comecem juntos) */
     [cite_start]MPI_Barrier(MPI_COMM_WORLD); [cite: 782-783, 786]
     tempo_inicio = MPI_Wtime(); // Ponto de inicio da medicao
 
     /* * FASE 1: DISTRIBUICAO (Setup)
      * Processo 0 (Mestre) distribui as linhas ciclicamente.
      */
     if (meu__rank == 0) {
         // Processo 0 copia os dados originais
         memcpy(Acpy, A, n * n * sizeof(double));
         memcpy(bcpy, b, n * sizeof(double));
 
         // Envia as linhas para os respectivos processos
         for (int i = 0; i < n; i++) {
             int rank_destino = i % total_de_processos;
             if (rank_destino != 0) { // Nao envia para si mesmo
                 [cite_start]MPI_Send(&Acpy[i * n], n, MPI_DOUBLE, rank_destino, i, MPI_COMM_WORLD); [cite: 331-347]
                 [cite_start]MPI_Send(&bcpy[i], 1, MPI_DOUBLE, rank_destino, i, MPI_COMM_WORLD); [cite: 331-347]
             }
         }
     } else {
         // Processos Escravos recebem suas linhas
         for (int i = meu__rank; i < n; i += total_de_processos) {
             [cite_start]MPI_Recv(&Acpy[i * n], n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status_recebimento); [cite: 374-392]
             [cite_start]MPI_Recv(&bcpy[i], 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status_recebimento); [cite: 374-392]
         }
     }
 
     /* * FASE 2: ELIMINACAO (Computacao Paralela)
      */
 
     // Buffer para receber a linha pivo em cada iteracao
     double *linha_pivo_buffer = (double *) malloc(n * sizeof(double));
     double valor_pivo_b; // Variavel para guardar o 'b' da linha pivo
 
     for (int i = 0; i < (n - 1); i++) {
 
         // --- Fase 2.A: Comunicacao (Broadcast da linha pivo) ---
         int rank_dono_pivo = i % total_de_processos;
 
         // O dono da linha pivo copia ela para o buffer
         if (meu__rank == rank_dono_pivo) {
             memcpy(linha_pivo_buffer, &Acpy[i * n], n * sizeof(double));
             valor_pivo_b = bcpy[i];
         }
 
         // O dono faz o Bcast da linha pivo e do elemento b pivo
         [cite_start]MPI_Bcast(linha_pivo_buffer, n, MPI_DOUBLE, rank_dono_pivo, MPI_COMM_WORLD); [cite: 813, 817-818, 827-832]
         [cite_start]MPI_Bcast(&valor_pivo_b, 1, MPI_DOUBLE, rank_dono_pivo, MPI_COMM_WORLD); [cite: 813, 817-818, 827-832]
 
         // --- Fase 2.B: Computacao (Cada processo atualiza suas linhas) ---
         for (int j = (i + 1); j < n; j++) {
 
             // Cada processo so atualiza as linhas que *pertencem* a ele
             if (meu__rank == (j % total_de_processos)) {
                 double ratio = Acpy[j * n + i] / linha_pivo_buffer[i]; // Usa pivo do buffer
                 for (int count = i; count < n; count++) {
                     Acpy[j * n + count] -= (ratio * linha_pivo_buffer[count]);
                 }
                 bcpy[j] -= (ratio * valor_pivo_b); // Usa pivo 'b' do buffer
             }
         }
 
         // (Opcional) Uma barreira aqui sincronizaria cada passo 'i'
         // MPI_Barrier(MPI_COMM_WORLD);
     }
     free(linha_pivo_buffer);
 
 
     /* * FASE 3: COLETA E RETRO-SUBSTITUICAO (Gather & Solve)
      * Os escravos enviam suas linhas processadas de volta ao mestre.
      * O Mestre (rank 0) executa a retro-substituicao sequencialmente.
      */
     if (meu__rank != 0) {
         // Escravos enviam suas linhas de volta
         for (int i = meu__rank; i < n; i += total_de_processos) {
             [cite_start]MPI_Send(&Acpy[i * n], n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD); [cite: 331-347]
             [cite_start]MPI_Send(&bcpy[i], 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD); [cite: 331-347]
         }
     } else {
         // Mestre recebe as linhas atualizadas
         for (int i = 1; i < n; i++) { // Comeca em 1 (linha 0 ja esta aqui)
             int rank_remetente = i % total_de_processos;
             if (rank_remetente != 0) {
                 [cite_start]MPI_Recv(&Acpy[i * n], n, MPI_DOUBLE, rank_remetente, i, MPI_COMM_WORLD, &status_recebimento); [cite: 374-392]
                 [cite_start]MPI_Recv(&bcpy[i], 1, MPI_DOUBLE, rank_remetente, i, MPI_COMM_WORLD, &status_recebimento); [cite: 374-392]
             }
         }
 
         /* Back-substitution (Execucao Serial no Rank 0) */
         /* O codigo aqui e identico ao original sequencial */
         x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
         for (int i = (n - 2); i >= 0; i--) {
             double temp = bcpy[i];
             for (int j = (i + 1); j < n; j++) {
                 temp -= (Acpy[i * n + j] * x[j]);
             }
             x[i] = temp / Acpy[i * n + i];
         }
     }
 
     /* Sincroniza antes de parar o tempo */
     [cite_start]MPI_Barrier(MPI_COMM_WORLD); [cite: 782-783, 786]
     tempo_fim = MPI_Wtime(); // Ponto final da medicao
 
     if (meu__rank == 0) {
         printf("------------------------------------------------------\n");
         printf("Tempo de execucao (solveLinearSystem): %f segundos\n", tempo_fim - tempo_inicio);
         printf("------------------------------------------------------\n");
     }
 
     /* Todos os processos liberam suas copias locais */
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