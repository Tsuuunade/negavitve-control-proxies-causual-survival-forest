library(grf)

cat("============================================================\n")
cat("Performance & Time Test: R GRF (No Proxies)\n")
cat("============================================================\n")

# Load data
obs_df = read.csv("data/latest_nc_synth.csv")
truth_df = read.csv("data/latest_nc_truth.csv")

# Extract variables (No Proxies!)
p_covariates = 5
x_cols = paste0("X", 0:(p_covariates-1))
X = as.matrix(obs_df[, x_cols])
A = obs_df$A
Y = obs_df$time
delta = obs_df$event
true_cate = truth_df$CATE_XU_eq7

# Determine RMST Horizon equivalent (using max as grf documentation usually uses, or 90th ptile)
# We will just run standard unrestricted target or standard RMST.
cat("\n--- Starting Model Training ---\n")
start_time = Sys.time()

# Train standard Causal Survival Forest with only X
model = causal_survival_forest(
  X = X,
  Y = Y,
  W = A,
  D = delta,
  horizon = max(Y),
  num.trees = 1000  # Give it enough trees to be fair
)

fit_time = Sys.time() - start_time
cat(sprintf("--- Training Complete in %.2f seconds ---\n", as.numeric(fit_time, units="secs")))

cat("\n--- Starting CATE Prediction ---\n")
pred_start_time = Sys.time()
cate_preds = predict(model, X)$predictions
pred_time = Sys.time() - pred_start_time
cat(sprintf("--- Prediction Complete in %.2f seconds ---\n", as.numeric(pred_time, units="secs")))

# Statistical Performance
bias = mean(cate_preds - true_cate)
mae = mean(abs(cate_preds - true_cate))
mse = mean((cate_preds - true_cate)^2)
rmse = sqrt(mse)
correlation = cor(cate_preds, true_cate)

cat("\n--- Detailed Statistical Performance (In-Sample) ---\n")
cat(sprintf("Mean Predicted CATE (RMST):  %.4f (StdDev: %.4f)\n", mean(cate_preds), sd(cate_preds)))
cat(sprintf("Mean True CATE (Unbounded):  %.4f (StdDev: %.4f)\n", mean(true_cate), sd(true_cate)))
cat(sprintf("Bias (Pred - True):          %.4f\n", bias))
cat(sprintf("Mean Absolute Error (MAE):   %.4f\n", mae))
cat(sprintf("Mean Squared Error (MSE):    %.4f\n", mse))
cat(sprintf("Root Mean Sq Error (RMSE):   %.4f\n", rmse))
cat(sprintf("Pearson Correlation:         %.4f\n", correlation))
cat("============================================================\n")