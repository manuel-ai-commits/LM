# Preamble

I am learning C! This is one of my first libraries to contribute to Machine Learning using low-level programming languages. I accept any kind of suggestions. Be kind!

# LM - Linear Model in C

A simple linear regression implementation in C with gradient descent. Does the job of fitting models to CSV data without any external ML libraries.

## What it does

Fits a linear model to your data using gradient descent. You get your standard regression metrics (R², Adjusted R², MSE, RSS) and can split your data into train/test sets. The example included predicts GPA from SAT scores.

## Dependencies

This uses two small libraries as git submodules:
- `libcsv` - for parsing CSV files
- `mat` - custom matrix operations library

Both are included in `deps/` and handle the heavy lifting for data loading and matrix math.

## Building

./build.shThat's it. The script updates submodules and compiles with `cc`. If you want to tweak compiler flags, they're in `build.sh`.

## Usage

The basic flow is:

1. Load your CSV data
2. Allocate a linear model with `lm_alloc(n_features)`
3. Initialize your data matrices with `init_data()`
4. Split into train/test if needed
5. Train with `lm_train()`
6. Get predictions with `lm_inf()`
7. Check your metrics

See `lm.c` for a working example. You can configure:
- `RESPONSE_COL` - which column is your response variable
- `HEADER` - whether there is the header or not
- `HEADER_ROW` - in which row the header is
- `SPLIT` - train/test split ratio
- `EPOCHS` - training iterations
- `LR` - learning rate

# Example run
Compile the C program, I use cc:
`cc -Wall -W -Wextra -O2 lm.c`
Or, even better, use my sh file `build.sh`, which also updates git submodules:
`sh build.sh`
Then run it specifying the csv location:
`./a.out data.csv`

## CSV Format

First row should be headers. Response variable (what you're predicting) goes in the column specified by `RESPONSE_COL`. Everything else becomes a feature. The code adds an intercept automatically.

## Metrics

After training you get:
- Residuals
- RSS (Residual Sum of Squares)
- TSS (Total Sum of Squares)
- MSE (Mean Squared Error)
- R² 
- Adjusted R²
- Degrees of Freedom

## Current State

Works fine for basic linear regression. There are a few TODOs in the code:
- Add graphical output
- Replace some loops with matrix

## License

Public domain.
