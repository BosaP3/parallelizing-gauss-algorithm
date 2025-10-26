# 1. Criar um arquivo de texto com o tamanho da matriz (N).
echo "7500" > entrada.txt

# 1. Compilar o código sequencial com a flag '-pg' para habilitar o profiling.
#    O executável será 'gauss_seq'.
gcc -pg -o gauss_seq gauss_fonte.c -lm

# 2. Executar o programa. Esta execução irá gerar um arquivo chamado 'gmon.out',
#    que contém os dados da análise.
./gauss_seq < entrada.txt

# 3. Usar o gprof para ler o executável e o 'gmon.out', gerando um relatório
#    de texto legível chamado 'analise_gprof.txt'.
gprof gauss_seq gmon.out > analise_gprof.txt

# 4. (Opcional) Visualizar o relatório gerado.
cat analise_gprof.txt
# Ou, para uma visualização paginada:
less analise_gprof.txt

---

# 1. Compilar o código paralelo ('gauss_paralelo.c').
#    A flag '-fopenmp' é essencial para ativar o OpenMP.
#    A flag '-O3' é uma otimização de compilação recomendada para testes de performance.
gcc -o gauss_par gauss_paralelo.c -fopenmp -lm

# 2. Executar o programa paralelo definindo o número de threads desejado.
#    Isso é feito com a variável de ambiente OMP_NUM_THREADS.

# Teste com 2 threads
OMP_NUM_THREADS=2 ./gauss_par < entrada.txt

# Teste com 4 threads
OMP_NUM_THREADS=4 ./gauss_par < entrada.txt

# Teste com 8 threads
OMP_NUM_THREADS=8 ./gauss_par < entrada.txt

# Teste com 16 threads
OMP_NUM_THREADS=16 ./gauss_par < entrada.txt