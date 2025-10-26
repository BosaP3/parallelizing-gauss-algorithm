### 1. Compile a versão MPI (usando mpicc)

```bash
mpicc gauss_mpi.c -o gauss_mpi -lm
```

### 2. Execute a versão MPI localmente
 O comando 'mpirun' executará o programa localmente por padrão.
 '-np 4' diz ao MPI para rodar com 4 processos.
 (O programa vai pedir o 'n', digite o MESMO valor de antes)

```bash
mpirun -np 4 ./gauss_mpi
```