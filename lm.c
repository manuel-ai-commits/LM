#include <time.h>

#define LM_IMPLEMENTATION
#include "lm.h"
    
#define EPOCHS 100000
#define LR 1e-04

int main () {
    srand(time(0));
    CSV_BUFFER *buf = csv_create_buffer();
    csv_load(buf, "data.csv");


    size_t n_feat = CSV_COLS(*buf, 0);
    LM lm = lm_alloc(n_feat);
    lm_rand(lm, 0, 1);

    Mat *data = init_data(lm, *buf);
    Mat X = data[0];
    Mat Y = data[1];
    Mat Y_hat = mat_alloc(1, Y.cols);
    metrics m_met = metrics_alloc(Y.cols);
    METRICS_PRINT(m_met);
    lm_train(&lm, &m_met, X, Y, LR, EPOCHS);
    METRICS_PRINT(m_met);
    // lm_inf(&Y_hat, lm, X);
    // MAT_PRINT(Y_hat);
    // MAT_PRINT(Y);



    // lm_print(lm, "STA_GPA");


    csv_destroy_buffer(buf);


    return 0;
}