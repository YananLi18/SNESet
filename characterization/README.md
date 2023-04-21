# Characterization
We characterize and compare the QoS and QoE metrics in SNESet with existing publicly available datasets and  ***qualitatively*** investigate the impact of QoS on QoE using Kendall correlation and relative information gain.

- **Kendall correlation**. Kendall correlation is a rank correlation that does not assume anything about the underlying distributions, noise, or nature of relationships. In contrast, Pearson correlation assumes Gaussian noise and a roughly linear relationship between variables. To compute the correlation, we bin stall ratios based on the QoS metrics, using bin sizes appropriate for each QoS metric of interest. The Kendall correlation coefficients are then computed between the mean-per-bin vector and the values of the bin indices for each application.
- **Information Gain**. Correlation provides a first-order summary of monotone relationships between stall ratio and QoS metrics. The information gain can corroborate or augment the correlation when the relationship is not monotone. The entropy of random variable Y and the conditional entropy of Y given another random variable X is defined as:

$$
\begin{aligned}
H(Y) & =-\sum_i P\left[Y=y_i\right] \log P\left[Y=y_i\right] \\
H(Y \mid X) & =\sum_j P\left[X=x_j\right] H\left(Y \mid X=x_j\right)
\end{aligned}
$$

where $𝑃 [𝑌 = 𝑦_𝑖]$ is the probability that $𝑌 = 𝑦_𝑖 $. And the relative information gain is $\frac{H(Y)-H(Y \mid X)}{H(Y)}$.

Folder `scripts` provide the scripts for plotting the figures (Figure 2 to Figure 11) in our paper.
