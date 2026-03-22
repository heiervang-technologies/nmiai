# Advanced Techniques for Minimizing KL Divergence in Astar Island

**Current best:** Bucket-based Bayesian template matching with per-round weighting, CV wKL = 0.028.
**Task:** 40x40 grid, 6 classes, entropy-weighted KL divergence scoring, 50 queries/round budget.

---

## 1. Dirichlet Posterior Averaging (HIGH PRIORITY)

**What:** Instead of predicting point-estimate class probabilities, place a Dirichlet prior on each cell's categorical distribution and update it with observations. The Dirichlet is the conjugate prior for categorical/multinomial data, so posterior updates are exact and closed-form.

**How it helps:** The Dirichlet posterior mean automatically provides calibrated probability predictions that minimize expected KL divergence under the posterior. For cells with no observations, the prior concentrations encode your template beliefs. For observed cells, the posterior concentrations shift toward observed frequencies. This is provably optimal under the Bayesian framework.

**Implementation:**
- For each cell, maintain alpha parameters (one per class): `alpha = [alpha_0, ..., alpha_5]`
- Prior: set alpha from historical bucket frequencies (your current template priors)
- Update: when you observe cell value k, increment `alpha_k += 1`
- Predict: `p_i = alpha_i / sum(alpha)` (posterior mean)
- The posterior mean minimizes expected KL divergence from the true distribution

**Key insight from literature:** Bayesian estimation of KL divergence for categorical systems using mixtures of Dirichlet priors (Camaglia & Nemenman, 2023) shows that mixture-of-Dirichlet priors capture multi-modal categorical distributions better than a single Dirichlet. This maps directly to your regime detection: each regime is a different Dirichlet prior.

**Expected gain:** Moderate. Your bucket priors are already an approximation of this; the gain comes from making the Bayesian update formal and composition-correct when mixing across regimes.

---

## 2. Temperature Scaling / Post-hoc Calibration (MEDIUM PRIORITY)

**What:** After computing raw probability predictions, apply a single temperature parameter T to the log-probabilities before renormalizing: `q_i = softmax(log(p_i) / T)`.

**How it helps:** Modern neural networks (and hand-tuned predictors) tend to be overconfident. Temperature scaling with T > 1 spreads probability mass more evenly, reducing KL divergence when the ground truth is more uncertain than your prediction suggests. Guo et al. (2017) showed this single-parameter calibration is "surprisingly effective."

**Implementation:**
- Compute raw predictions p for all cells
- Apply: `q_i = exp(log(p_i) / T) / Z` where Z normalizes
- Tune T on your 90+ ground truth maps using cross-validated wKL
- Different T values for different cell buckets (distance-to-civ, terrain type)

**Variant - Per-bucket temperature:** Instead of one global T, learn separate temperatures for each bucket (coastal cells, inland cells near settlements, forest cells, etc.). This is analogous to "class-wise calibration" from the literature.

**Expected gain:** Small-to-moderate. You already use tau, but per-bucket tuning could squeeze out 0.002-0.005 wKL.

---

## 3. Label Smoothing / Probability Floor Optimization (MEDIUM PRIORITY)

**What:** Instead of a fixed 0.01 floor for all classes, optimize the floor/smoothing parameter per class and per cell type. Label smoothing replaces sharp distributions with a mixture: `q_smoothed = (1-alpha)*q + alpha*uniform`.

**How it helps:** KL divergence is asymmetric and heavily penalizes q_i = 0 when p_i > 0. The optimal floor depends on how often each class appears as a surprise. For example, ruins (class 3) might appear unexpectedly at 2% frequency - your floor should be close to 2%, not 1%.

**Implementation:**
- For each bucket, compute the empirical frequency of each class across all ground truth maps
- Set the floor to max(empirical_freq * 0.5, 0.005) per class per bucket
- This ensures rare-but-possible outcomes get enough probability mass
- Optimize alpha in label smoothing using CV on ground truth

**Expected gain:** Small but reliable. Especially important for high-entropy cells where surprises are common.

---

## 4. Mixture of Historical Round Templates with Bayesian Model Averaging (HIGH PRIORITY)

**What:** Instead of selecting a single regime or template, maintain a probability distribution over all historical round templates and predict using the weighted mixture.

**How it helps:** Bayesian Model Averaging (BMA) is provably optimal for minimizing expected KL divergence when the true model is in your set. Even when it is not, BMA typically outperforms model selection because it hedges against regime misidentification.

**Implementation:**
```
P(class | cell) = sum_t w_t * P_t(class | cell)
```
where w_t = P(template t | observations) computed via Bayes rule:
```
w_t propto P(observations | template t) * P(template t)
```

**Key insight:** Your current approach already does something like this, but the gain comes from:
1. Using the full posterior over templates (not argmax)
2. Updating weights continuously as more observations arrive
3. Allowing the mixture to naturally hedge between regimes

**Expected gain:** Moderate. This is your highest-leverage theoretical improvement since regime misidentification is your biggest error source.

---

## 5. Spatial Markov Random Field Regularization (MEDIUM-HIGH PRIORITY)

**What:** Add spatial coherence to predictions by modeling neighboring cells as dependent. A Markov Random Field (MRF) encodes the prior belief that adjacent cells tend to have correlated outcomes.

**How it helps:** Your current predictor treats each cell independently (given its bucket). But in reality, settlements cluster, ruins form connected regions, and forest patches are contiguous. An MRF can smooth predictions spatially, reducing noise in cell-by-cell estimates.

**Implementation:**
- Define a pairwise potential: neighboring cells with similar predictions get higher probability
- Use loopy belief propagation or mean-field variational inference to solve
- Tune the coupling strength on ground truth maps
- Only apply to non-static cells (exclude mountains, fixed ocean)

**Simpler version:** Gaussian smoothing of the probability maps with a small kernel (sigma=0.5-1.0), then renormalize. This is a rough approximation to MRF smoothing but much simpler to implement.

**Expected gain:** Small-to-moderate. Most useful for cells at the boundary between settlement expansion zones and wilderness.

---

## 6. Learned Transition Rules from Replay Trajectories (HIGH PRIORITY)

**What:** Use your 260+ replay trajectories to learn the exact cellular automaton transition rules, then run forward simulation to predict final distributions.

**How it helps:** If you can learn the transition kernel P(next_state | current_state, neighborhood), you can simulate the CA forward from the initial map and compute empirical probability distributions over final states. This directly optimizes for the ground truth generation process.

**Implementation:**
- From replays, extract (cell_state, neighbor_states, next_cell_state) tuples
- Fit a lookup table or small neural network for P(next | current, neighbors)
- Run Monte Carlo forward simulation: start from initial map, apply learned rules stochastically, repeat 1000+ times
- The empirical frequency of each class at each cell IS your probability prediction

**Key insight from ARC-AGI 2025:** The winning NVARC approach used test-time training on structured puzzle descriptions. Similarly, your replays ARE the training data for learning the simulator's rules.

**Expected gain:** High if the rules are learnable and the simulation is faithful. This is the most direct approach since it models the data-generating process.

---

## 7. Conformal Predictive Distributions (LOW-MEDIUM PRIORITY)

**What:** Use conformal prediction to produce calibrated probability distributions with coverage guarantees. Unlike Bayesian methods, conformal prediction is distribution-free.

**How it helps:** Conformal Predictive Distributions (CPDs) output an entire CDF per prediction with guaranteed calibration. They can detect when your model is uncertain and automatically widen the distribution.

**Implementation:**
- Split ground truth maps into calibration and test sets
- For each cell, compute nonconformity scores based on prediction errors
- Use the nonconformity distribution to produce calibrated probability intervals
- Convert intervals to probability distributions over classes

**Expected gain:** Low for this specific problem since you already have strong Bayesian priors. More useful if your model were black-box.

---

## 8. Entropy-Weighted Loss Optimization (HIGH PRIORITY)

**What:** Since scoring uses entropy-weighted KL divergence, optimize your predictions directly for this metric rather than raw KL.

**How it helps:** High-entropy cells (where the ground truth distribution is spread across many classes) contribute more to your score. Allocating more modeling effort (and query budget) to these cells has higher ROI than improving predictions on low-entropy cells.

**Implementation:**
- Estimate per-cell entropy from your priors (cells near settlements have high entropy; ocean/mountain cells have low entropy)
- Weight your temperature scaling, smoothing, and query allocation by estimated entropy
- For high-entropy cells: use more conservative (flatter) predictions
- For low-entropy cells: use more confident predictions
- Allocate more of your 50 queries to observe high-entropy regions

**Key insight:** If a cell's ground truth entropy is H, the KL contribution is weighted by H. Reducing KL by 0.01 on a high-entropy cell (H=1.5) is worth 3x more than on a low-entropy cell (H=0.5).

**Expected gain:** Moderate-to-high. This reframes the entire optimization around the actual scoring metric.

---

## 9. Ensemble of Predictors with Stacking (MEDIUM PRIORITY)

**What:** Combine multiple prediction methods (bucket priors, template matching, learned simulator, spatial model) using stacked generalization. Train a meta-learner to weight predictions from each base predictor.

**How it helps:** Different predictors excel on different cell types. A stacked ensemble can learn which predictor to trust for which cell type, achieving better overall performance than any single method.

**Implementation:**
- Generate predictions from N base methods on held-out ground truth maps
- Train a meta-model (logistic regression or small neural net) that takes base predictions + cell features as input and outputs the optimal mixture weights
- The meta-model can be trained directly to minimize wKL on the validation set

**Key insight from literature:** Using stacking to average Bayesian predictive distributions (Yao et al., 2018) shows that stacking outperforms simple BMA when models are misspecified, which they always are in practice.

**Expected gain:** Moderate. Requires multiple diverse base predictors to be effective.

---

## 10. Variational Inference with Structured Prior (LOW PRIORITY)

**What:** Use variational Bayesian inference to learn a structured latent variable model of the simulation outcomes. The latent variable captures the "regime" and the observation model captures per-cell distributions.

**How it helps:** This generalizes your template matching by learning a continuous latent space of simulation outcomes rather than discretizing into a fixed number of regimes.

**Implementation:**
- Define generative model: z ~ Prior (regime latent), x_ij | z ~ Categorical(f(z, cell_features))
- Use variational inference to approximate P(z | observations)
- Predict: P(x_ij) = integral P(x_ij | z) P(z | obs) dz
- Can be implemented as a VAE trained on ground truth maps

**Expected gain:** Low. Requires significant engineering for marginal improvement over discrete templates given limited training data (90 maps).

---

## 11. Refinement Loops / Test-Time Adaptation (MEDIUM-HIGH PRIORITY)

**What:** Iteratively refine predictions using feedback from observations. Inspired by ARC-AGI 2025 where "refinement loops" were the defining winning theme.

**How it helps:** Instead of a single forward pass from priors to predictions, iterate: predict -> observe -> update -> re-predict. Each iteration uses the latest posterior to choose the most informative next query.

**Implementation:**
- Start with template priors
- Query 1-2 seeds to identify regime (information gain maximization)
- Update regime posterior across all 5 seeds
- For remaining budget, use active learning to query cells with highest expected entropy reduction
- After each observation, re-run the full prediction pipeline with updated posteriors

**Expected gain:** Moderate. Your current approach already does some of this; the gain comes from formalizing the active learning query selection.

---

## 12. Monte Carlo Forward Simulation from Partial Observations (HIGH PRIORITY)

**What:** Use initial maps + learned transition rules to run many Monte Carlo simulations forward. Use observations to reject simulations that are inconsistent with observed cell values, keeping only consistent trajectories.

**How it helps:** This is Approximate Bayesian Computation (ABC). By conditioning on observations, you narrow the distribution of possible outcomes. The surviving simulations give you a more accurate probability distribution.

**Implementation:**
- Run N=10000 forward simulations from initial map
- When you observe a cell value at step 50, reject all simulations where that cell has a different value
- The surviving simulations define your posterior predictive distribution
- This naturally handles spatial correlations since each simulation is a complete 40x40 grid

**Key advantage:** This captures spatial correlations that cell-independent methods miss. If you observe a large settlement cluster in one area, the surviving simulations will correctly predict correlated outcomes in nearby cells.

**Expected gain:** High if forward simulation is faithful. This is the gold standard for probabilistic prediction of stochastic processes.

---

## 13. Isotonic Regression Calibration (MEDIUM PRIORITY)

**What:** Instead of parametric calibration (temperature scaling), use isotonic regression to learn a non-parametric monotone mapping from predicted probabilities to calibrated probabilities.

**How it helps:** Isotonic regression makes no assumptions about the functional form of miscalibration. It learns the optimal monotone transformation from predictions to calibrated probabilities, which can capture non-linear miscalibration that temperature scaling misses.

**Implementation:**
- For each class, sort cells by predicted probability
- Use isotonic regression to fit the mapping: predicted_p -> empirical_frequency
- Apply the learned mapping to all predictions
- Renormalize to ensure probabilities sum to 1

**Expected gain:** Small-to-moderate. Most useful if your predictions have non-linear miscalibration.

---

## 14. Gaussian Process Spatial Interpolation (LOW-MEDIUM PRIORITY)

**What:** Use Gaussian Processes to interpolate observed probabilities across the grid, accounting for spatial correlation.

**How it helps:** When you observe a few cells, a GP can predict what nearby unobserved cells look like, with calibrated uncertainty. The GP posterior naturally provides probability distributions.

**Implementation:**
- Define a spatial kernel (Matern or RBF) over cell coordinates
- For each class, fit a GP to the observed cell probabilities
- Use GP posterior mean and variance to construct calibrated predictions at unobserved cells
- The GP naturally handles the exploration-exploitation tradeoff in query selection

**Expected gain:** Low. Your bucket-based approach already captures most spatial structure through cell features (distance-to-civ, terrain type).

---

## 15. Direct wKL Optimization via Gradient Descent (MEDIUM-HIGH PRIORITY)

**What:** Instead of manually tuning parameters, directly optimize the entropy-weighted KL divergence on your ground truth validation set using gradient descent.

**How it helps:** The wKL objective is differentiable with respect to the predicted probabilities. You can backpropagate through your entire prediction pipeline (template weights, temperatures, floors, smoothing parameters) to find the globally optimal configuration.

**Implementation:**
- Parameterize your prediction as: q(cell) = softmax((1/T_bucket) * log(sum_t w_t * prior_t(cell)))
- Make T_bucket, w_t, and floor parameters differentiable
- Compute wKL on ground truth maps
- Use Adam optimizer to minimize wKL
- Use cross-validation to prevent overfitting (90 maps is enough for ~20 parameters)

**Expected gain:** Moderate-to-high. This is the most direct path to squeezing out the last few points, since you are literally optimizing the scoring metric.

---

## Prioritized Recommendations

### Tier 1: Implement First (highest expected gain, lowest risk)
1. **Direct wKL optimization** (#15) - directly optimize the scoring metric
2. **Entropy-weighted query allocation** (#8) - focus budget on what matters
3. **Bayesian Model Averaging over templates** (#4) - hedge against regime misidentification

### Tier 2: Implement If Time Permits
4. **Monte Carlo forward simulation with ABC rejection** (#12) - captures spatial correlations
5. **Learned transition rules from replays** (#6) - model the data-generating process
6. **Refinement loops** (#11) - iterative prediction improvement
7. **Spatial MRF smoothing** (#5) - spatial coherence

### Tier 3: Diminishing Returns
8. **Dirichlet posterior averaging** (#1) - formalize what you already do
9. **Per-bucket temperature scaling** (#2) - fine-grained calibration
10. **Label smoothing optimization** (#3) - floor optimization
11. **Isotonic regression** (#13) - non-parametric calibration
12. **Ensemble stacking** (#9) - combine diverse predictors

### Tier 4: High Effort, Low Marginal Gain
13. **Conformal prediction** (#7) - distribution-free guarantees
14. **Gaussian Process interpolation** (#14) - spatial interpolation
15. **Variational inference** (#10) - continuous latent regimes
