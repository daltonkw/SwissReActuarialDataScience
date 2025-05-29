# This implementation provides:
# Key Features:

# Attention Mechanism: Learns feature-specific routing weights (0-1) for each input feature
# Dual Pathways: Linear path for simple relationships, non-linear path for complex patterns
# Feature Weighting: Each feature gets routed based on learned attention weights
# Actuarial Integration: Works with both exposure offset and GLM predictions (CANN style)

# How It Works:

# High attention weight (→1): Feature routed to non-linear path
# Low attention weight (→0): Feature routed to linear path
# Mixed routing: Features can partially use both paths

# Keys:

# Feature-level attention: Each input feature gets its own routing decision
# Conservative initialization: Starts with preference for linear relationships
# Dropout regularization: Prevents overfitting in non-linear path
# Analysis tools: Functions to interpret which features need non-linear modeling



# Attention-Based Feature Routing for Poisson Frequency Model
# This model learns which features need non-linear treatment vs linear relationships

# Define input layers
Design   <- keras::layer_input(shape = c(q0),  dtype = 'float32', name = 'Design')
VehBrand <- keras::layer_input(shape = c(1),   dtype = 'int32', name = 'VehBrand')
Region   <- keras::layer_input(shape = c(1),   dtype = 'int32', name = 'Region')
LogVol   <- keras::layer_input(shape = c(1),   dtype = 'float32', name = 'LogVol')

# Create embeddings for categorical variables
BrandEmb <- VehBrand |> 
  keras::layer_embedding(
    input_dim = VehBrandLabel, 
    output_dim = d, 
    name = 'BrandEmb'
  ) |> 
  keras::layer_flatten(name='Brand_flat')

RegionEmb <- Region |> 
  keras::layer_embedding(
    input_dim = RegionLabel, 
    output_dim = d, 
    name = 'RegionEmb'
  ) |> 
  keras::layer_flatten(name='Region_flat')

# Combine all features (excluding LogVol for now)
all_features <- keras::layer_concatenate(
  list(Design, BrandEmb, RegionEmb), 
  name = 'all_features'
)


# Calculate total feature dimension for attention
total_feature_dim <- q0 + 2*d  # Design features + 2 embeddings

# ==============================================================================
# ATTENTION MECHANISM: Learns which features need non-linear treatment
# ==============================================================================
attention_weights <- all_features |>
  keras::layer_dense(units = 64, activation = 'relu', name = 'attention_hidden1') |>
  keras::layer_dense(units = 32, activation = 'relu', name = 'attention_hidden2') |>
  keras::layer_dense(
    units = total_feature_dim, 
    activation = 'sigmoid', 
    name = 'attention_weights',
    kernel_initializer = keras::initializer_constant(0.1),  # Start with slight preference for linear
    bias_initializer = keras::initializer_constant(-0.5)    # Bias toward linear initially
  )

# ==============================================================================
# LINEAR PATH: Traditional GLM-style linear relationships
# ==============================================================================
linear_path <- all_features |>
  keras::layer_dense(
    units = 1, 
    activation = 'linear', 
    name = 'linear_path',
    kernel_initializer = keras::initializer_random_normal(stddev = 0.01),
    bias_initializer = keras::initializer_constant(log(lambda_hom))
  )

# ==============================================================================
# LINEAR PATH: Traditional GLM-style linear relationships
# ==============================================================================
linear_path <- all_features |>
  keras::layer_dense(
    units = 1, 
    activation = 'linear', 
    name = 'linear_path',
    kernel_initializer = keras::initializer_random_normal(stddev = 0.01),
    bias_initializer = keras::initializer_constant(log(lambda_hom))
  )

# ==============================================================================
# NON-LINEAR PATH: Deep network for complex relationships
# ==============================================================================
nonlinear_path <- all_features |>
  keras::layer_dense(units = q1, activation = 'tanh', name = 'nonlinear_hidden1') |>
  keras::layer_dropout(rate = 0.2, name = 'dropout1') |>
  keras::layer_dense(units = q2, activation = 'tanh', name = 'nonlinear_hidden2') |>
  keras::layer_dropout(rate = 0.2, name = 'dropout2') |>
  keras::layer_dense(units = q3, activation = 'tanh', name = 'nonlinear_hidden3') |>
  keras::layer_dense(
    units = 1, 
    activation = 'linear', 
    name = 'nonlinear_path',
    kernel_initializer = keras::initializer_zeros(),  # Start conservative
    bias_initializer = keras::initializer_constant(log(lambda_hom))
  )

# ==============================================================================
# FEATURE-WISE ATTENTION ROUTING
# ==============================================================================
# We need to apply attention at the feature level, so we'll use a custom approach
# Create weighted features for each path

linear_weights <- attention_weights |> 
  keras::layer_lambda(
    f = function(x) 1 - x,
    name = "linear_weights"
  )

# For linear path: apply (1 - attention) weighting
lin_multiply_layer <- keras::keras$layers$Multiply(name = "linear_weighted_features")
linear_weighted_features <- lin_multiply_layer(list(all_features, linear_weights))

# For nonlinear path: apply attention weighting
nonlin_multiply_layer <- keras::keras$layers$Multiply(name = "nonlinear_weighted_features")
nonlinear_weighted_features <- nonlin_multiply_layer(list(all_features, attention_weights))

# Recompute paths with weighted features
linear_output <- linear_weighted_features |>
  keras::layer_dense(
    units = 1, 
    activation = 'linear', 
    name = 'linear_output',
    kernel_initializer = keras::initializer_random_normal(stddev = 0.01),
    bias_initializer = keras::initializer_constant(log(lambda_hom))
  )

nonlinear_output <- nonlinear_weighted_features |>
  keras::layer_dense(units = q1, activation = 'tanh', name = 'nl_hidden1') |>
  keras::layer_dropout(rate = 0.2) |>
  keras::layer_dense(units = q2, activation = 'tanh', name = 'nl_hidden2') |>
  keras::layer_dropout(rate = 0.2) |>
  keras::layer_dense(units = q3, activation = 'tanh', name = 'nl_hidden3') |>
  keras::layer_dense(
    units = 1, 
    activation = 'linear', 
    name = 'nonlinear_output',
    kernel_initializer = keras::initializer_zeros(),
    bias_initializer = keras::initializer_constant(log(lambda_hom))
  )

# ==============================================================================
# COMBINE PATHS
# ==============================================================================
# Combine linear and nonlinear outputs
comb_add_layer <- keras::keras$layers$Add(name = "combined_paths")

combined_network <- comb_add_layer(list(linear_output, nonlinear_output))

# Add the offset (exposure or GLM predictions)
eta_add_layer <- keras::keras$layers$Add(name = "eta")
eta <- eta_add_layer(list(combined_network, LogVol))

# Apply exponential for Poisson
response <- eta |> 
  keras::layer_activation(activation = "exponential", name = "Response")

# ==============================================================================
# CREATE MODEL
# ==============================================================================
model_attention <- keras::keras_model(
  inputs = list(Design, VehBrand, Region, LogVol), 
  outputs = response,
  name = "AttentionBasedFeatureRouting"
)

# ==============================================================================
# COMPILE MODEL
# ==============================================================================
model_attention$compile(
  loss = "poisson",
  optimizer = keras::optimizer_nadam(learning_rate = 0.001),
)

# Print model summary
summary(model_attention$summary())

# ==============================================================================
# TRAINING EXAMPLE
# ==============================================================================
# For traditional offset approach:
# Vtrain <- as.matrix(log(train$Exposure))

# For CANN approach with GLM predictions:
# Vtrain <- as.matrix(log(train$fitGLM2))
# lambda_hom <- sum(train$ClaimNb) / sum(train$fitGLM2)

# Training call would be:
# attention_fit <- model_attention |> keras::fit(
#   x = list(Xtrain, VehBrandtrain, Regiontrain, Vtrain),
#   y = as.matrix(train$ClaimNb),
#   epochs = 300L,
#   batch_size = 10000L,
#   validation_data = list(
#     list(Xvalid, VehBrandvalid, Regionvalid, Vvalid),
#     as.matrix(valid$ClaimNb)
#   ),
#   verbose = 1L,
#   callbacks = list(
#     keras::callback_early_stopping(
#       monitor = "val_loss",
#       patience = 20,
#       restore_best_weights = TRUE
#     ),
#     keras::callback_reduce_lr_on_plateau(
#       monitor = "val_loss",
#       factor = 0.5,
#       patience = 10
#     )
#   )
# )

# ==============================================================================
# ATTENTION ANALYSIS FUNCTIONS
# ==============================================================================

# Function to extract and analyze attention weights
analyze_attention <- function(model, sample_data) {
  # Create a model that outputs attention weights
  attention_model <- keras::keras_model(
    inputs = model$input,
    outputs = model$get_layer('attention_weights')$output
  )
  
  # Get attention weights for sample data
  attention_weights <- predict(attention_model, sample_data)
  
  # Create feature names (you'll need to adjust based on your actual feature names)
  feature_names <- c(
    paste0("Design_", 1:q0),
    paste0("Brand_emb_", 1:d),
    paste0("Region_emb_", 1:d)
  )
  
  # Summary statistics
  attention_summary <- data.frame(
    feature = feature_names,
    mean_attention = colMeans(attention_weights),
    std_attention = apply(attention_weights, 2, sd)
  )
  
  # Sort by mean attention (features needing most non-linear treatment)
  attention_summary <- attention_summary[order(-attention_summary$mean_attention), ]
  
  return(list(
    weights = attention_weights,
    summary = attention_summary
  ))
}

# Usage example:
# attention_analysis <- analyze_attention(
#   model_attention, 
#   list(Xtrain[1:1000,], VehBrandtrain[1:1000], Regiontrain[1:1000], Vtrain[1:1000,])
# )
# print(attention_analysis$summary)