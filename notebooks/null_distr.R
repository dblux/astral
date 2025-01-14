compute_t <- function(x, mu = 0) {
  n <- length(x)
  t <- (mean(x) - mu) / (sd(x) / sqrt(n))
  t
}


x_arr <- abs(replicate(1000, rnorm(5, 0, 1)))
tstats <- apply(x_arr, 2, compute_t)

hist(tstats, breaks = 40, freq = FALSE)

curve(2 * dt(x, df = 9), add = T)

hist(abs(rnorm(1000, 0, 1)), breaks = 50)

# Null hypothesis: Paired differences are normally distributed with mean 0 but with what variance?
#   -> abs(paired diff): half-normal distribution (if var is 1)
#   ->

library(ggplot2)
library(dplyr)

data(mtcars)
head(mtcars)

# calculate mean mpg for cyl using tidyverse
mtcars %>%
  group_by(cyl) %>%
  summarize(mean_mileage = mean(mpg), mean_wt = mean(wt))

ggplot(mtcars, aes(x = cyl, y = mpg, col = cyl)) +
  geom_point(position = position_jitter(width = 0.2))
