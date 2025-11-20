//TODO: Understand why with low LR it goes to hell, with 1e-04 is ok LOL
//TODO: Change for to matrix dot products
//TODO: Case in which the Y and X are in different CSVs
//TODO: Change the LM struct and split it in two structs
// This is a header guard. It prevents multiple inclusion of this header file during compilation.
#ifndef LM_H_
#define LM_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef LM_MALLOC
#include <stdlib.h>
#define LM_MALLOC malloc
#endif 

#ifndef LM_ASSERT
#include <assert.h>
#define LM_ASSERT assert
#endif 

#ifndef RESPONSE_COL
#define RESPONSE_COL 0
#endif 

#ifndef HEADER_ROW
#define HEADER_ROW 0
#endif

#ifndef HEADER
#define HEADER true
#endif


#define CSV_IMPLEMENTATION
#include "libcsv/csv.h"

#define MAT_IMPLEMENTATION
#include "mat.h"


typedef struct {
    char **name_features;      
    char *name_response;       
    size_t n_features;         
    Mat *B;
} LM;

typedef struct {
    Mat *residuals;
    double r_s;
    double adj_r_s;
    double rss;
    double tss;
    double mse;
    size_t dof;
} metrics;


#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])

LM lm_alloc(size_t n_feat);
metrics metrics_alloc(size_t n_feat);
Mat *init_data(LM lm, CSV_BUFFER buf);
void lm_print(LM lm, const char *name);
void metrics_print(metrics m, const char *name);
#define LM_PRINT(lm) lm_print(lm, #lm); // A macro for prinitng the whole structure of a neural network
#define METRICS_PRINT(m) metrics_print(m, #m); // A macro for prinitng the whole structure of a neural network
void LM_rand(LM lm, float low, float high);
void lm_inf(Mat *Y_hat, LM lm, Mat X);
void grad_betas(Mat *grad, Mat X, Mat Y, Mat Y_hat);
void lm_loss(metrics *m_met, Mat X, Mat Y, Mat Y_hat);
void lm_train(LM *lm, metrics *m_met, Mat X, Mat Y, float lr, size_t epochs);
/* Comment for now
void nn_zero(lm lm);
float nn_loss(lm lm, Mat in, Mat out);
void nn_finite_diff(lm lm, lm g, float eps, Mat in, Mat out);
void nn_backprop(lm lm, lm g, Mat in, Mat out);
void nn_learn(lm lm, lm g, float rate);
*/
#endif // NN_H_



// C part
#ifdef LM_IMPLEMENTATION
/*
    Given an array with each neuron per layer, including the initial one
    and the number of layers in total, including the input.

    It returns a completely allocated neural network structure with weights
    and biases correctly formatted.
*/
LM lm_alloc(size_t n_feat) {
    LM lm;

    lm.B = LM_MALLOC(sizeof(*lm.B));
    LM_ASSERT(lm.B != NULL);
    lm.name_features = LM_MALLOC(sizeof(*lm.name_features));
    LM_ASSERT(lm.name_features != NULL);
    lm.name_response = LM_MALLOC(sizeof(*lm.name_response));
    LM_ASSERT(lm.name_response != NULL);
    *lm.B = mat_alloc(1, n_feat);
    return lm;
}

metrics metrics_alloc(size_t n_feat) {
    metrics m_met;
    m_met.residuals = LM_MALLOC(sizeof(*m_met.residuals));
    LM_ASSERT(m_met.residuals != NULL);

    *m_met.residuals = mat_alloc(1, n_feat);
    return m_met;
}

//TODO: Maybe swap from checking header to checking heaer row if present
Mat *init_data(LM lm, CSV_BUFFER buf) {
    //TODO: ASSERT WIDTHS ARE THE SAME
    size_t temp = CSV_COLS(buf, 0);
    for (size_t i = 1; i < CSV_ROWS(buf); i++) {
        assert(CSV_COLS(buf, i) == temp);
        temp = CSV_COLS(buf, i);
    }

    // ============ FILL THE NAMES IN LM ============
    // Fill the response 
    if (HEADER){
        lm.name_response = CSV_ENTRY(buf, HEADER_ROW, RESPONSE_COL);
    }
    csv_remove_field(&buf, HEADER_ROW, RESPONSE_COL);
    // Fill the features names
    lm.name_features[0] = "Intercept";
    if (HEADER) {
        for (size_t j = 0; j < CSV_COLS(buf, HEADER_ROW); j++) {
            lm.name_features[j+1] =  CSV_ENTRY(buf, HEADER_ROW, j);
        }
    } 
    csv_remove_row(&buf, HEADER_ROW);
    
    // ============ FILL Y ============
    Mat Y = mat_alloc(1, CSV_ROWS(buf));
    // Fill the response variable
    for (size_t i = 0; i < CSV_ROWS(buf); i++) {
        MAT_AT(Y, 0, i) = atof(CSV_ENTRY(buf, i, RESPONSE_COL));
    }
    // Delete the field already allocated, to not have problems during the other loops
    csv_remove_col(&buf, RESPONSE_COL);
    
    // ============ FILL X ============
    Mat X = mat_alloc(CSV_ROWS(buf), CSV_COLS(buf,0)+1);
    // FILL THE INTERCEPT
    for (size_t i = 0;i < CSV_ROWS(buf); i++) {
        MAT_AT(X, i, 0) = 1.0;
    }

    for (size_t i = 0;i < CSV_ROWS(buf); i++) {
        for (size_t j = 0; j < CSV_COLS(buf, i); j++) {
            MAT_AT(X, i, j+1) = atof(CSV_ENTRY(buf, i, j));
        }
    }
    Mat *out = MAT_MALLOC(2 * sizeof(Mat));
    if (!out) {
        return NULL;
    }
    out[0] = X;
    out[1] = Y;
    return out;
}


/*
    Print float values with the same formatting as lm_print and metrics_print
*/
void print_value_f(float value, const char *name){
    printf("%*s%s = %f,\n", (int) 4, "", name, value);
}
/*
    Print int unsigned values with the same formatting as lm_print and metrics_print
*/
void print_value_zu(size_t value, const char *name){
    printf("%*s%s = %zu,\n", (int) 4, "", name, value);
}
/*
    Print the LM parameters
*/
void lm_print(LM lm, const char *name) {
    printf("%s = [\n", name);
    mat_print(*lm.B, "B", 4);
    printf("]\n");
}

/*
    Print the metrics results
*/
void metrics_print(metrics m_met, const char *name) {
    printf("%s = [\n", name);
    mat_print(*m_met.residuals, "Residuals", 4);
    print_value_f(m_met.rss, "RSS");
    print_value_f(m_met.tss, "TSS");
    print_value_f(m_met.mse, "MSE");
    print_value_f(m_met.r_s, "R^2");
    print_value_f(m_met.adj_r_s, "Adj R^2");
    print_value_zu(m_met.dof, "Degrees of Freedom");
    printf("]\n");
}

/*
    Randomize the betas
*/
void lm_rand(LM lm, float low, float high) {
    mat_rand(*lm.B, low, high);
}

/*
    Compute the predicted response 
*/
void lm_inf(Mat *Y_hat, LM lm, Mat X) {
    assert(X.cols == lm.B->cols);
    mat_zeros(*Y_hat);
    for (size_t i = 0; i < X.rows; i++){
        for (size_t j = 0; j < X.cols; j++){
            MAT_AT(*Y_hat, 0, i) += MAT_AT(*lm.B, 0, j) * MAT_AT(X, i, j);
        }
    }
}

void lm_loss(metrics *m_met, Mat X, Mat Y, Mat Y_hat) {
    // Ensure the sizes are equal, asserting separately for clarity
    LM_ASSERT(m_met->residuals->cols == Y.cols);
    LM_ASSERT(Y.cols == Y_hat.cols);
    Mat res = mat_alloc(Y.rows, Y.cols);
    mat_copy(res, Y);

    // initialize n and p
    size_t n = X.rows;
    size_t p = X.cols;
    assert(n != 0);

    // COMPUTE RESIDUALS
    mat_nsum(res, Y_hat);
    mat_copy(*m_met->residuals, res);
    // COMPUTE RSS
    mat_copy(res, Y);
    Mat RSS = mat_SS(res, Y_hat);
    assert(RSS.rows == 1 && RSS.cols == 1);
    m_met->rss = MAT_AT(RSS, 0, 0);
    // COMPUTE MSE
    m_met->mse = m_met->rss/n;
    // COMPUTE TSS
    mat_copy(res, Y);
    float Y_mean = MAT_AT(mat_row_sum(res), 0, 0)/n;
    Mat SS = mat_SS_const(res, Y_mean);
    assert(SS.rows == 1 && SS.cols == 1);
    m_met->tss = MAT_AT(SS, 0, 0);
    mat_release(&res);
    // Compute R^2
    assert(m_met->tss != 0);
    m_met->r_s = 1.0 - (m_met->rss/m_met->tss);
    // Compute Degrees of Freedoom
    m_met->dof = n - p - 1;
    // Compute Adjusted R^2
    assert(m_met->dof != 0);
    m_met->adj_r_s = 1.0 - (1.0 - m_met->r_s) * ((double)(n - 1) / (double)m_met->dof);
}

void grad_betas(Mat *grad, Mat X, Mat Y, Mat Y_hat){
    Mat res = mat_alloc(Y.rows, Y.cols);
    mat_copy(res, Y);
    mat_nsum(res, Y_hat);

    mat_dot(*grad, res, X);
    mat_prod_const(*grad, 2.0f / Y.cols);
    mat_release(&res);
}

void lm_train(LM *lm, metrics *m_met, Mat X, Mat Y, float lr, size_t epochs){
    Mat Y_hat = mat_alloc(1, Y.cols);
    Mat grad = mat_alloc(1, lm->B->cols);


    for(size_t i = 0; i<epochs; i++){
        // Make inference
        lm_inf(&Y_hat, *lm, X);
        // Compute gradient of the betas
        grad_betas(&grad, X, Y, Y_hat);
        // Apply learning rate
        mat_prod_const(grad, lr);
        // Compute Gradient Descend
        mat_sum(*lm->B, grad);
    }
    // Compute loss
    lm_loss(m_met, X, Y, Y_hat);
}
/* TO INTEGRATE



*/

/* COMMENTED FOR NOW


void nn_finite_diff(lm lm, lm g, float eps, Mat in, Mat out) {
    // In this implementation, both weights and biases are updated by finite difference.
    // However, you might observe that after repeated epochs, the finite difference for weights stays at 0,
    // while the biases keep changing (if the loss surface is flat with respect to weights, gradients will be 0).
    float saved;

    float l_ = nn_loss(lm, in, out);

    for (size_t l = 0; l < lm.n_layers; ++l) {
        // Compute finite difference for weights
        for (size_t i = 0; i < lm.ws[l].rows; ++i) {
            for (size_t j = 0; j < lm.ws[l].cols; ++j) {
                saved = MAT_AT(lm.ws[l], i, j);
                MAT_AT(lm.ws[l], i, j) += eps;
                float grad = (nn_loss(lm, in, out) - l_) / eps;
                MAT_AT(g.ws[l], i, j) = grad;
                MAT_AT(lm.ws[l], i, j) = saved;
            }
        }

        // Compute finite difference for biases
        for (size_t i = 0; i < lm.bs[l].rows; ++i) {
            for (size_t j = 0; j < lm.bs[l].cols; ++j) {
                saved = MAT_AT(lm.bs[l], i, j);
                MAT_AT(lm.bs[l], i, j) += eps;
                float grad = (nn_loss(lm, in, out) - l_) / eps;
                MAT_AT(g.bs[l], i, j) = grad;
                MAT_AT(lm.bs[l], i, j) = saved;
            }
        }
    }
}

void nn_backprop(lm lm, lm g, Mat in, Mat out) {
    LM_ASSERT(in.rows == in.rows);
    size_t n = in.rows;
    LM_ASSERT(NN_OUTPUT(lm).cols == out.cols);

    nn_zero(g); 
    // 'i' - Current sample 
    // 'l' - Current layer
    // 'j' - Current neuron activation
    // 'k' - Previous neuron activation
    
    for (size_t i = 0; i < n; ++i) {
        mat_copy(NN_INPUT(lm), mat_row(in, i));
        nn_forward(lm);
        
        for (size_t j = 0; j <= lm.n_layers; ++j) {
            mat_fill(g.as[j], 0);
        }

        for (size_t j= 0; j < out.cols ; ++j) {
            MAT_AT(NN_OUTPUT(g), 0, j) = 2*(MAT_AT(NN_OUTPUT(lm), 0, j) - MAT_AT(out, i, j));
        }

        for (size_t l = lm.n_layers; l > 0 ; --l) {
            for (size_t j = 0; j < lm.as[l].cols; ++j) {
                float a = MAT_AT(lm.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                MAT_AT(g.bs[l-1], 0, j) += da*a*(1 - a);
                for (size_t k = 0; k < lm.as[l-1].cols; ++k) {
                    // 'j' - weight matrix column
                    // 'k' - weight matrix row
                    float pa = MAT_AT(lm.as[l-1], 0, k);
                    float w = MAT_AT(lm.ws[l-1], k, j);
                    MAT_AT(g.ws[l-1], k, j) += da*a*(1 - a)*pa;
                    MAT_AT(g.as[l-1], 0, k) += da*a*(1 - a)*w;
                }
            }
        }
    }
    for (size_t i = 0; i< g.n_layers; ++i) {
        for (size_t j = 0; j< g.ws[i].rows; ++j) {
            for (size_t k = 0; k< g.ws[i].cols; ++k) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }

        for (size_t k = 0; k< g.bs[i].cols; ++k) {
            MAT_AT(g.bs[i], 0, k) /= n;
        }
        
    }
}



void nn_learn(lm lm, lm g, float rate) {
    // The learning step is identical for both weights and biases.
    // If the gradients (g.ws) for weights remain 0 over epochs, weights do not change,
    // but if g.bs is nonzero, the biases keep updating.
    for (size_t l = 0; l < lm.n_layers; ++l) {
        for (size_t i = 0; i < lm.ws[l].rows; ++i) {
            for (size_t j = 0; j < lm.ws[l].cols; ++j) {
                MAT_AT(lm.ws[l], i, j) -= MAT_AT(g.ws[l], i, j) * rate;
            }
        }
        for (size_t i = 0; i < lm.bs[l].rows; ++i) {
            for (size_t j = 0; j < lm.bs[l].cols; ++j) {
                MAT_AT(lm.bs[l], i, j) -= MAT_AT(g.bs[l], i, j) * rate;
            }
        }
    }
}
*/
#endif // NN_IMPLEMENTATION