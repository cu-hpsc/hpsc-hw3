# HW3: Domain decomposition and stationary iteration
## Due: 2019-10-14 (Monday)

Click to [accept this assignment](https://classroom.github.com/a/nq25J0Vt)

## Building PETSc

```
$ wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.12.0.tar.gz
```

```
$ tar xf petsc-lite-3.12.0.tar.gz
$ cd petsc-3.12.0
$ export PETSC_DIR=$HOME/petsc-3.12.0 PETSC_ARCH=mpich-dbg
$ cd $PETSC_DIR
$ ./configure --with-fortran-bindings=0 --with-blaslapack-lib=/opt/conda/lib/liblapack.so
[...]
$ make
```

Whenever you open a new terminal (or log back in), you'll need to export the
`PETSC_DIR` and `PETSC_ARCH` environment variables; you can run

```
$ echo 'export PETSC_DIR=$HOME/petsc-3.12.0 PETSC_ARCH=mpich-dbg' >> .bash_profile
```

to add them to your default environment.

### Optimized version

The debugging version is preferable for development, but you'll want an
optimized build to measure performance.  You can get that by configuring with a
different `PETSC_ARCH`.

```
$ export PETSC_ARCH=mpich-opt
$ ./configure --with-fortran-bindings=0 --with-blaslapack-lib=/opt/conda/lib/liblapack.so --with-debugging=0 COPTFLAGS='-O3 -march=native"
[...]
$ make
```

Now when you compile code, you can select the optimized version via
```
$ make PETSC_ARCH=mpich-opt solve
```


## Building and running the example

Clone this repository and run `make`.  You should be able to execute using
`./solve`.  To see all relevant run-time options, use

    $ ./solve -help

There will be a section describing the options defined in the example
```
Poisson solver -------------------------------------------------
  -omega <1. : 1.>: Relaxation factor for iterative method (None)
  -max_it <now 100 : formerly 100>: Maximum number of iterations (None)
```

You can also run in parallel via

    $ mpiexec -n 2 ./solve

Check `lscpu` or `lstopo` or to determine the number of physical and virtual
cores you have available.

## Part 1: Jacobi scaling

The example currently implements Richardson iteration, which depends on an
important tunable parameter called `omega`.  Add a feature to the code to use
the [Jacobi iterative method](https://en.wikipedia.org/wiki/Jacobi_method).

* Does the parameter `omega` that yields fastest convergence change as you refine the grid?

* Does the convergence history depend on the number of processes you use?

* Use
  [`PetscLogStagePush()`](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Profiling/PetscLogStagePush.html)
  and related functions to profile the important computation within each
  iteration.  Run with `-log_view` to see what PETSc says about performance.
  (In particular, it will distinguish the neighbor communication (see
  `VecScatter` events) from norm computation (`VecNorm`; uses `MPI_Allreduce`).
  Can you use this information to tune the performance (e.g., to converge
  faster)?

* Add OpenMP parallelism.  How does the performance of OpenMP parallelism
  compare with that of MPI?
  
  - You may want to consider [process binding](https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager#Process-core_Binding)
    and OpenMP parameters such as `OMP_NUM_THREADS` and [`OMP_PROC_BIND`](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fPROC_005fBIND.html#OMP_005fPROC_005fBIND).

* As a function of grid resolution (on square grids, `-da_grid_x 30 -da_grid_y 30`),
  how does the number of iterations to reduce the residual by 1e-3 change?

  - Make a plot (actual data and/or using an analytic model) in terms that are
    relevant to a user who believes that approximation error scales like
    `1/n^2`.  How does cost (e.g., CPU seconds) to solve scale with increased
    accuracy requirements?
    
  - Make a plot in terms that are relevant to a user with access to a large
    parallel resource and external requirements on time.  How does attainable
    accuracy scale with allowed execution time?  What about cost/efficiency?

#### Workflow tip

I would suggest using Python, R, or Julia to parse output into a data frame
inside `Report.ipynb`.  You can make the output from the C code easy to parse
(e.g., write CSV format).  Think about what information you might need in
advance.  You could create new targets in the `Makefile` to manage running
experiments to update log files.

## Part 2: [Gauss-Seidel](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method)

Implement a Gauss-Seidel method using lexicographic ordering (like the existing
loops).  Is this easy to parallelize?  There is a variant called Block Jacobi,
in which each process updates its own local patch using Gauss-Seidel, but does
not update across process boundaries until completing the local patches.  (This
is probably what your algorithm will do if you naively implement Gauss-Seidel.)

The Gauss-Seidel update can be implemented in-place using
`DMLocalToLocalBegin()`/`DMLocalToLocalEnd()`, but you may need to implement the
norm yourself.

* Does this method have the same convergence history if you change the number of
  processes?
  
* How does this method compare with Jacobi in terms of time and cost (cf. plots
  you made in part 1)?

## Part 3: Red-black Gauss-Seidel

One variant of Gauss-Seidel for the 5-point stencil "colors" the grid like a
checkerboard and updates all red points, communicates on the boundaries, then
updates all black points.  Implement this method and compare to the previous
methods.

### Part 4: Reproducibility

Would another class member be able to read your report and reproduce your
results?  Revisit the above parts to ensure that this is indeed achievable.  In
`Report.ipynb`, document any additional information that may be needed to
reproduce and explain what obstacles you anticipate that may prevent
reproduction.
