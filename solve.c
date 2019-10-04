static char help[] = "Solve the Poisson problem using the finite difference method with a DMDA.\n\n";

/*
  The discretization uses the 5-point stencil

      -1
  -1   4   -1
      -1

  to solve the Poisson problem iteratively.  To compute the Laplacian (-u_xx - u_yy), we would normally divide by h^2.
  To keep this operator scaled uniformly with respect to mesh size, we instead multiply the whole thing (including the
  right-hand side) by h^2.  If L is the Laplacian, we wish to solve

    L u = 1

  This type of problem (with arbitrary right-hand side) is called the Poisson problem.  We impose that the solution
  u(x,y) go to zero at the boundaries.

*/

#include <petscdm.h>
#include <petscdmda.h>

/* Compute V = 1 - Laplacian(U) */
static PetscErrorCode Residual(DM dm, Vec U, Vec V) {
  PetscErrorCode ierr;
  const PetscScalar **u;
  PetscScalar **v;
  DMDALocalInfo info;
  Vec Xlocal;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(dm, &Xlocal);CHKERRQ(ierr); // A work vector
  ierr = DMGlobalToLocal(dm, U, INSERT_VALUES, Xlocal);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(dm, Xlocal, &u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm, V, &v);CHKERRQ(ierr);
  PetscReal h2 = 1. / ((info.mx + 1) * (info.my + 1));
  for (PetscInt j=info.ys; j<info.ys+info.ym; j++) {
    for (PetscInt i=info.xs; i<info.xs+info.xm; i++) {
      // laplacian is a linear function of the input state u
      PetscScalar laplacian = 4*u[j][i] - u[j][i-1] - u[j][i+1] - u[j-1][i] - u[j+1][i];
      v[j][i] = h2 - laplacian;
    }
  }
  ierr = DMDAVecRestoreArrayRead(dm, Xlocal, &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm, V, &v);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Xlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode   ierr;
  DM               dm;
  Vec              U, V;
  PetscReal        omega;
  PetscInt         max_it;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsSetValue(NULL, "-draw_cmap", "plasma");CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Poisson solver","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-omega","Relaxation factor for iterative method",NULL,omega=0.1,&omega,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-max_it","Maximum number of iterations",NULL,max_it=100,&max_it,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, // GHOSTED reserves padding on the boundary (which we leave as zero)
                      DMDA_STENCIL_STAR,                        // STAR stencil means we only need neighbors along grid lines, not corners
                      20, 20,                                   // Default dimensions of grid; can set with -da_grid_x 30 -da_grid_y 30
                      PETSC_DECIDE,PETSC_DECIDE,
                      1,        // degrees of freedom at each grid point
                      1,        // grid distance to neighbors needed in residual evaluation
                      NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&V);CHKERRQ(ierr);

  for (PetscInt it=0; it<max_it; it++) {
    ierr = Residual(dm, U, V);CHKERRQ(ierr);
    {
      PetscScalar norm;
      ierr = VecNorm(V, NORM_2, &norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Iteration %D: norm %g\n", it, norm);CHKERRQ(ierr);
    }
    ierr = VecAXPY(U, omega, V);CHKERRQ(ierr); // u_{n+1} = u_n + omega * (b - L u_n)
  }

  // If you have X11, you can run with -solution_view draw to visualize the solution
  ierr = VecViewFromOptions(U, NULL, "-solution_view");CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
