#include <time.h>

//TODO: Just use header row to assert wheteher there is an header or not
#define HEADER_ROW 0
#define HEADER true
#define RESPONSE_COL 0

#define LM_IMPLEMENTATION
#include "lm.h"

#define SPLIT 0.9
    
#define EPOCHS 10000000
#define LR 1e-04


int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <csv_file>\n", argv[0]);
        return 1;
    }
    srand(time(0));
    CSV_BUFFER *buf = csv_create_buffer();
    csv_load(buf, argv[1]);


    size_t n_feat = CSV_COLS(buf, 0);
    LM *lm = lm_alloc(n_feat);

    Mat **data = init_data(lm, buf);
    Mat **X_split = mat_split_rows(data[0], SPLIT);
    Mat **Y_split = mat_split_cols(data[1], SPLIT);

    
    /* ================ DESCEND METHOD ================ */
#if 0
    Mat *X_train = X_split[0];
    Mat *X_test  = X_split[1];
    Mat *Y_train = Y_split[0];
    Mat *Y_test = Y_split[1];

    metrics *met_train = metrics_alloc(Y_train->cols);
    lm_train(lm, met_train, X_train, Y_train, LR, EPOCHS);
    METRICS_PRINT(met_train);
    // // Prediction
    Mat *Y_hat = mat_alloc(1, Y_test->cols);
    metrics *met_test = metrics_alloc(Y_test->cols);
    lm_inf(Y_hat, lm, X_test);
    lm_loss(met_test, X_test, Y_test, Y_hat);
    METRICS_PRINT(met_test);   

#else
    /* ================ OLS METHOD ================ */
    Mat *X = data[0];
    Mat *Y = data[1];
    OLS(lm, X, Y);
    MAT_PRINT(lm->B);
    Mat *Y_hat = mat_alloc(1, Y->cols);
    metrics *met_train = metrics_alloc(Y->cols);
    lm_inf(Y_hat, lm, X);
    lm_loss(met_train, X, Y, Y_hat);
    METRICS_PRINT(met_train);


#endif
    csv_destroy_buffer(buf);

    return 0;
}