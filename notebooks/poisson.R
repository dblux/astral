x <- 0:10
y <- dpois(x, 1)
print(y)
barplot(y, names.arg = x)
