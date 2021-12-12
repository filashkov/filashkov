/* Include benchmark-specific header. */
#include "heat-3d.h"
#include <omp.h>
#include <iostream>
#include <vector>

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

double bench_t_start, bench_t_end;

static double
rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0) {
        printf("Error return from gettimeofday: %d", stat);
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void
bench_timer_start()
{
    bench_t_start = rtclock();
}

void
bench_timer_stop()
{
    bench_t_end = rtclock();
}

void
old_bench_timer_print()
{
    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

void
bench_timer_print()
{
    printf("%0.6lf\n", bench_t_end - bench_t_start);
}


static void
init_array(vector<vector<vector<double>>>& A, vector<vector<vector<double>>>& B)
{
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                A[i][j][k] = B[i][j][k] = (double)(i + j + (n - k)) * 10 / (n);
            }
        }
    }
}

static void
print_array(const vector<vector<vector<double>>>& A)
{
    int n = A.size();
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "A");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if ((i * n * n + j * n + k) % 20 == 0) {
                    fprintf(stderr, "\n");
                }
                fprintf(stderr, "%0.2lf ", A[i][j][k]);
            }
        }
    }
    fprintf(stderr, "\nend   dump: %s\n", "A");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void
kernel_heat_3d(int tsteps, vector<vector<vector<double>>>& A,
        vector<vector<vector<double>>>& B)
{
    int n = A.size();
    for (int t = 1; t <= TSTEPS; t++) {
#pragma omp parallel for collapse(3)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    B[i][j][k] = 0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k])
                            + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k])
                            + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1])
                            + A[i][j][k];
                }
            }
        }
#pragma omp parallel for collapse(3)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    A[i][j][k] = 0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k])
                            + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k])
                            + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1])
                            + B[i][j][k];
                }
            }
        }
    }
}


void
local_main(int threads_num, int n, int tsteps = TSTEPS)
{
    omp_set_dynamic(0); // отключение динамической настройки количество потоков
    omp_set_num_threads(threads_num); // установка числа тредов

    vector<vector<vector<double>>> A(n, vector<vector<double>>(n, vector<double>(n, 0)));
    vector<vector<vector<double>>> B(n, vector<vector<double>>(n, vector<double>(n, 0)));

    init_array(A, B);

    bench_timer_start();

    kernel_heat_3d(tsteps, A, B);

    bench_timer_stop();
    bench_timer_print();
}

int
main()
{
    vector<int> threads_num = { 4, 8, 16, 32, 64, 128, 256, 512 };
    vector<int> data_size = { 5, 10, 20, 40, 80, 120, 200, 300 };

    for (int i = 0; i < threads_num.size(); i++) {
        for (int j = 0; j < data_size.size(); j++) {
            cout << threads_num[i] << " " << data_size[j] << " ";
            local_main(threads_num[i], data_size[j]);
        }
    }

    return 0;
}
