#include "Residual.cuh"

namespace cfd {
void steady_screen_output(int step, real err_max, gxl::Time &time, std::array<real, 4> &res) {
  time.get_elapsed_time();
  FILE *history = std::fopen("history.dat", "a");
  fprintf(history, "%d\t%11.4e\n", step, err_max);
  fclose(history);

  printf("\n%38s    converged to: %11.4e\n", "rho", res[0]);
  printf("  n=%8d,                       V     converged to: %11.4e   \n", step, res[1]);
  printf("  n=%8d,                       p     converged to: %11.4e   \n", step, res[2]);
  printf("%38s    converged to: %11.4e\n", "T ", res[3]);
  printf("CPU time for this step is %16.8fs\n", time.step_time);
  printf("Total elapsed CPU time is %16.8fs\n", time.elapsed_time);
}

void
unsteady_screen_output(int step, real err_max, gxl::Time &time, std::array<real, 4> &res, real dt, real solution_time) {
  time.get_elapsed_time();
  FILE *history = std::fopen("history.dat", "a");
  fprintf(history, "%d\t%11.4e\n", step, err_max);
  fclose(history);

  printf("\n%38s    converged to: %11.4e\n", "rho", res[0]);
  printf("  n=%8d,   dt=%13.7e,   V     converged to: %11.4e   \n", step, dt, res[1]);
  printf("  n=%8d,   dt=%13.7e,   p     converged to: %11.4e   \n", step, dt, res[2]);
  printf("%38s    converged to: %11.4e\n", "T ", res[3]);
  printf("Current physical  time is %16.8es\n", solution_time);
  printf("CPU time for this step is %16.8fs\n", time.step_time);
  printf("Total elapsed CPU time is %16.8fs\n", time.elapsed_time);
}
} // cfd