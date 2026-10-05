/* Busy CPU threads for the power rows in docs/building.md: each thread runs
 * AVX-512 FMAs (AVX2 without AVX-512) in bursts of duty% of every period,
 * all threads on the same period grid, and sleeps the rest. At 100% duty it
 * never sleeps.
 *
 * Build: gcc -O2 -march=native -pthread -o target/cpu-burn tools/cpu-burn.c
 * Usage: cpu-burn <threads> [duty%=100] [period_ms=100] [seconds=60]
 * It prints the busy core-seconds and the FMA rate when it ends, also when
 * it is stopped early with SIGINT or SIGTERM.
 */
#include <immintrin.h>
#include <signal.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double duty = 1.0, period = 0.1, seconds = 60.0, t0;
static volatile float sink;
static volatile sig_atomic_t quit;

static void on_signal(int sig) { (void)sig; quit = 1; }

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sleep_until(double t) {
    struct timespec ts;
    ts.tv_sec = (time_t)t;
    ts.tv_nsec = (long)((t - (double)ts.tv_sec) * 1e9);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
}

typedef struct { double busy; double fmas; } Result;

#ifdef __AVX512F__
typedef __m512 vec;
#define LANES 16
#define SET1 _mm512_set1_ps
#define FNMA _mm512_fnmadd_ps
#define ADD _mm512_add_ps
#define SUM(v) _mm512_reduce_add_ps(v)
#else
typedef __m256 vec;
#define LANES 8
#define SET1 _mm256_set1_ps
#define FNMA _mm256_fnmadd_ps
#define ADD _mm256_add_ps
static float SUM(__m256 v) { float f[8]; _mm256_storeu_ps(f, v); return f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7]; }
#endif

static void *burn(void *arg) {
    Result *r = arg;
    /* x = 1.9 - x*x is chaotic on [-1.96, 1.96], so the bits keep toggling. */
    vec c = SET1(1.9f);
    vec x0 = SET1(0.1f), x1 = SET1(0.2f), x2 = SET1(0.3f), x3 = SET1(0.4f),
        x4 = SET1(-0.1f), x5 = SET1(-0.2f), x6 = SET1(-0.3f), x7 = SET1(-0.4f);
    double end = t0 + seconds, busy = 0, rounds = 0;
    for (double start = t0; start < end && !quit; start += period) {
        double stop = start + (duty >= 1.0 ? period : period * duty);
        if (stop > end) stop = end;
        double t = now();
        if (t < start) { sleep_until(start); t = now(); }
        double begin = t;
        /* The clock is HPET on this laptop (1.4 us a call), so read it
         * after 8,192 FMAs per chain (about 20 us). */
        while (t < stop && !quit) {
            for (int i = 0; i < 8192; i++) {
                x0 = FNMA(x0, x0, c); x1 = FNMA(x1, x1, c); x2 = FNMA(x2, x2, c); x3 = FNMA(x3, x3, c);
                x4 = FNMA(x4, x4, c); x5 = FNMA(x5, x5, c); x6 = FNMA(x6, x6, c); x7 = FNMA(x7, x7, c);
            }
            rounds += 1;
            t = now();
        }
        busy += t - begin;
    }
    sink = SUM(ADD(ADD(ADD(x0, x1), ADD(x2, x3)), ADD(ADD(x4, x5), ADD(x6, x7))));
    r->busy = busy;
    r->fmas = rounds * 8192 * 8 * LANES;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <threads> [duty%%=100] [period_ms=100] [seconds=60]\n", argv[0]);
        return 2;
    }
    int threads = atoi(argv[1]);
    if (argc > 2) duty = atof(argv[2]) / 100.0;
    if (argc > 3) period = atof(argv[3]) / 1000.0;
    if (argc > 4) seconds = atof(argv[4]);
    if (threads < 1 || duty <= 0 || period <= 0 || seconds <= 0) {
        fprintf(stderr, "threads, duty, period and seconds must be positive\n");
        return 2;
    }
    pthread_t *ids = calloc(threads, sizeof *ids);
    Result *results = calloc(threads, sizeof *results);
    struct sigaction sa = {0};
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    double begin = now();
    t0 = begin + 0.01;
    for (int i = 0; i < threads; i++) pthread_create(&ids[i], NULL, burn, &results[i]);
    double busy = 0, fmas = 0;
    for (int i = 0; i < threads; i++) {
        pthread_join(ids[i], NULL);
        busy += results[i].busy;
        fmas += results[i].fmas;
    }
    printf("cpu-burn: %d threads at %.0f%% of %.0f ms for %.1f s: %.1f busy core-seconds, %.1f G FMA lanes/s per busy core\n",
           threads, duty * 100, period * 1000, now() - begin, busy, busy > 0 ? fmas / busy / 1e9 : 0);
    return 0;
}
