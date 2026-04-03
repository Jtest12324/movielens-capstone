################################################################
# HarvardX PH125.9x Data Science: Capstone
# MovieLens Project
# Author: Joy Roy (Jtest12324)
################################################################

# Install and load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# Download MovieLens 10M dataset
options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, files = c("ml-10M100K/ratings.dat", "ml-10M100K/movies.dat"))

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies_file <- "ml-10M100K/movies.dat"
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Create edx and final_holdout_test sets (provided by course)
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, removed, movielens)

# Feature engineering: Extract year from timestamp and title
edx <- edx %>%
  mutate(
    year_rated = year(as_datetime(timestamp)),
    year_released = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\))"))
  )

final_holdout_test <- final_holdout_test %>%
  mutate(
    year_rated = year(as_datetime(timestamp)),
    year_released = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\))"))
  )

# Create train/validation split from edx for model tuning
# NOTE: final_holdout_test is NEVER used for training or tuning
set.seed(1, sample.kind = "Rounding")
val_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-val_index,]
temp_val <- edx[val_index,]

validation <- temp_val %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed2 <- anti_join(temp_val, validation)
train_set <- rbind(train_set, removed2)

rm(val_index, temp_val, removed2)

# RMSE function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Model 1: Naive mean
mu <- mean(train_set$rating)
naive_rmse <- RMSE(validation$rating, mu)
cat("Naive RMSE:", naive_rmse, "\n")

rmse_results <- tibble(Method = "Naive Mean", RMSE = naive_rmse)

# Model 2: Movie effect
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

movie_rmse <- RMSE(validation$rating, predicted_ratings)
cat("Movie Effect RMSE:", movie_rmse, "\n")
rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie Effect", RMSE = movie_rmse))

# Model 3: Movie + User effects
user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

movie_user_rmse <- RMSE(validation$rating, predicted_ratings)
cat("Movie + User Effect RMSE:", movie_user_rmse, "\n")
rmse_results <- bind_rows(rmse_results, tibble(Method = "Movie + User Effects", RMSE = movie_user_rmse))

# Model 4: Regularized Movie + User effects
# Tune lambda parameter
lambdas <- seq(3, 7, 0.25)

rmses_lambda <- sapply(lambdas, function(l) {
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + l))
  
  predicted <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  RMSE(validation$rating, predicted)
})

lambda_opt <- lambdas[which.min(rmses_lambda)]
cat("Optimal lambda:", lambda_opt, "\n")

# Train final regularized model with optimal lambda
b_i_reg <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda_opt))

b_u_reg <- train_set %>%
  left_join(b_i_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda_opt))

predicted_ratings <- validation %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

reg_rmse <- RMSE(validation$rating, predicted_ratings)
cat("Regularized Movie + User RMSE:", reg_rmse, "\n")
rmse_results <- bind_rows(rmse_results, tibble(Method = "Regularized Movie + User", RMSE = reg_rmse))

# Model 5: Add Genre effect
genre_avgs <- train_set %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

predicted_ratings <- validation %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(b_g = replace_na(b_g, 0),
         pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

genre_rmse <- RMSE(validation$rating, predicted_ratings)
cat("Regularized Movie + User + Genre RMSE:", genre_rmse, "\n")
rmse_results <- bind_rows(rmse_results, tibble(Method = "Reg Movie + User + Genre", RMSE = genre_rmse))

# Model 6: Add Year Released effect
year_avgs <- train_set %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(b_g = replace_na(b_g, 0)) %>%
  group_by(year_released) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g))

predicted_ratings <- validation %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  left_join(year_avgs, by = "year_released") %>%
  mutate(b_g = replace_na(b_g, 0),
         b_y = replace_na(b_y, 0),
         pred = mu + b_i + b_u + b_g + b_y) %>%
  pull(pred)

year_rmse <- RMSE(validation$rating, predicted_ratings)
cat("Regularized Movie + User + Genre + Year RMSE:", year_rmse, "\n")
rmse_results <- bind_rows(rmse_results, tibble(Method = "Reg Movie + User + Genre + Year", RMSE = year_rmse))

print(rmse_results)

# Final model: Train on FULL edx, test on final_holdout_test
# Use best model (Model 6) trained on ALL of edx
mu_final <- mean(edx$rating)

b_i_final <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_final) / (n() + lambda_opt))

b_u_final <- edx %>%
  left_join(b_i_final, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_final - b_i) / (n() + lambda_opt))

b_g_final <- edx %>%
  left_join(b_i_final, by = "movieId") %>%
  left_join(b_u_final, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_final - b_i - b_u))

b_y_final <- edx %>%
  left_join(b_i_final, by = "movieId") %>%
  left_join(b_u_final, by = "userId") %>%
  left_join(b_g_final, by = "genres") %>%
  mutate(b_g = replace_na(b_g, 0)) %>%
  group_by(year_released) %>%
  summarize(b_y = mean(rating - mu_final - b_i - b_u - b_g))

# Predict on final_holdout_test
final_predictions <- final_holdout_test %>%
  left_join(b_i_final, by = "movieId") %>%
  left_join(b_u_final, by = "userId") %>%
  left_join(b_g_final, by = "genres") %>%
  left_join(b_y_final, by = "year_released") %>%
  mutate(b_g = replace_na(b_g, 0),
         b_y = replace_na(b_y, 0),
         pred = mu_final + b_i + b_u + b_g + b_y) %>%
  pull(pred)

# Calculate final RMSE
final_rmse <- RMSE(final_holdout_test$rating, final_predictions)

cat("\n================================\n")
cat("FINAL RMSE ON HOLDOUT TEST SET:", final_rmse, "\n")
cat("================================\n")
