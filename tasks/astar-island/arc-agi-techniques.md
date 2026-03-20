# ARC-AGI Techniques for Astar Island

## Bottom line

ARC techniques are useful here, but not as a plug-and-play solver.

The transferable idea is not "use an ARC solver on our grids". The transferable idea is to use the same **search over structured hypotheses** playbook:

- a compact transformation DSL,
- strong spatial priors,
- object-centric features,
- symmetry-aware data augmentation and canonicalization,
- and a refinement loop that proposes candidate rules, simulates them, and keeps only candidates that match the observed outputs.

For Astar Island, this should be adapted from **exact deterministic grid transformation** into **probabilistic local-rule induction** over repeated cellular automaton updates.

## Important correction on ARC-AGI-3

As of **March 20, 2026**, ARC-AGI-3 is an **interactive** benchmark, not the classic static input-grid to output-grid format, and ARC Prize stated in December 2025 that it planned to release ARC-AGI-3 in early 2026. That means ARC-AGI-3 does **not** yet provide the same kind of mature "winning static grid transformation" recipe that ARC-AGI-1 and ARC-AGI-2 do. For our problem, ARC-AGI-1/2 methods are the relevant reference class.

## 1. What the top ARC approaches were

### ARC 2020 to early ARC-AGI-1: brute-force DSL search

The 2020 winner (`icecuber`) used brute-force discrete program search and reached about **20%** on the private evaluation set. The ARC Prize 2024 report describes the pre-2024 ARC landscape as dominated by discrete search plus progressively better DSL design. This matters because it established the basic winning template: solve the task by searching over small interpretable programs rather than by pure end-to-end prediction.

What transfers:

- Small, interpretable rule languages can beat black-box prediction when the task is really about hidden transformations.
- Strong priors about objectness, geometry, counting, and neighborhood structure matter more than raw scale.

### ARC Prize 2024: three major families

The official 2024 technical report says progress re-accelerated around three categories:

- **Deep-learning-guided program synthesis**
- **Test-time training (TTT) / transductive models**
- **Hybrid systems combining synthesis and transduction**

The most practically relevant 2024 systems were:

1. **the ARChitects**

   - Won the 2024 open-source score prize at **53.5%** private eval.
   - Combined synthetic training data (`Re-ARC`, `Concept-ARC`, `ARC-Heavy`), ARC-specific tokenization, test-time fine-tuning, candidate generation, candidate scoring, and broad augmentation.
   - Explicitly used **D8 symmetry operations** (rotations and reflections), color permutations, and example reordering as augmentations.

2. **MindsAI / test-time training line of work**

   - The MIT TTT paper reports **53.0%** on public validation with an 8B LM and **61.9%** when ensembled with program-synthesis methods.
   - The key claim is that temporarily adapting the model to the specific puzzle at inference time is far better than treating each task as a frozen one-shot prompt.

3. **Induction + transduction hybrids**

   - The 2024 paper winner, *Combining Induction and Transduction for Abstract Reasoning*, found that the two modes solve different task types.
   - Induction/program synthesis helps with exact compositional reasoning; transduction helps with fuzzier perceptual pattern matching.

4. **Neurally guided program induction / latent-program search**

   - Ouellette's work frames ARC search as learning over grid space, program space, or transform space.
   - Bonnet and Macfarlane's Latent Program Network (LPN) replaces explicit DSL enumeration with search in a compact latent program space.

### ARC Prize 2025 / ARC-AGI-2: the year of the refinement loop

The official ARC Prize 2025 report says the defining theme of 2025 was the **refinement loop**: propose a candidate solution, verify it, use the error signal to improve the next candidate, and repeat.

The main systems were:

1. **NVARC**

   - Won ARC Prize 2025 on ARC-AGI-2 with **24.03%** private-eval score.
   - Combined a **multi-stage synthetic data pipeline**, an improved **ARChitects-style test-time-trained model**, and **Tiny Recursive Model** components.
   - NVIDIA's writeup summarizes the recipe as **synthetic data + test-time training + disciplined engineering**.

2. **the ARChitects (2025 version)**

   - Second place on ARC-AGI-2.
   - Shifted toward a more 2D-aware recursive self-refinement pipeline, still centered on per-task adaptation.

3. **Evolutionary / refinement-based program synthesis**

   - The 2025 paper awards highlighted **SOAR**, **evolutionary program synthesis**, and related generate-verify-refine loops.
   - These systems are especially interesting for Astar because they optimize programs against a verifier rather than trying to predict outputs directly.

4. **Tiny Recursive Models / zero-pretraining puzzle-specific learners**

   - TRM and CompressARC show that very small models trained at test time can do nontrivial ARC reasoning.
   - The main lesson is not the exact architecture. The lesson is that **per-task adaptation** can be stronger than a static global model when each task has its own hidden rules.

## 2. Which ARC techniques apply best to Astar Island?

## High transfer

### A. DSL-based rule search

This is the single best ARC idea to transfer.

Our problem is almost literally: infer a compact hidden transformation process from repeated examples of input grid -> output grid distribution.

What to copy from ARC:

- Use a **small domain-specific language** for update rules.
- Search over **compositions of local predicates** instead of opaque networks.
- Keep the rules **interpretable** so we can inspect whether they look like Growth, Conflict, Trade, Winter, and Environment.

For Astar Island the DSL should not be generic ARC primitives like "paint object red". It should be a CA DSL built around local terrain and neighborhood predicates, for example:

- `is_coastal(cell)`
- `count_neighbors(type in {...}, radius=r)`
- `distance_to_nearest(type)`
- `same_connected_component_as_initial_settlement`
- `if forest and adjacent_to_settlement then p_settlement += a`
- `if coastal and near_settlement then p_port += b`
- `if isolated_settlement then p_empty += c`
- `if ruin or abandoned then p_forest += d`

The critical adjustment is that our target is **probabilistic**, not deterministic. So the DSL should produce either:

- per-phase probabilistic updates, or
- local logits/energies for each class, which are normalized into probabilities.

### B. Refinement loops with a verifier

This is the second most useful ARC idea.

ARC 2025 showed that repeated **generate -> verify -> refine** loops are a strong pattern. For Astar Island the verifier is much cleaner than ARC because we have 25 full supervised examples.

A good verifier could score candidate rule sets using:

- KL divergence to the observed 40x40x6 probability tensors,
- argmax accuracy on final class maps,
- calibration on uncertain frontier cells,
- and regularization favoring short, local, reusable rules.

That suggests a search loop like:

1. Propose a rule program.
2. Run it for 50 years on all 25 maps.
3. Compare simulated class histograms or probabilities to ground truth.
4. Mutate the rule set or its parameters.
5. Keep changes that improve score.

This is much closer to ARC 2025 program refinement than to standard supervised learning.

### C. Neural pattern matching as a proposal model, not the final source of truth

This also transfers, but mainly as a helper.

ARC winners used neural models to:

- propose candidate programs,
- rank candidate outputs,
- or adapt to a specific task at test time.

For Astar Island, a neural model can do three useful things:

1. Predict final class probabilities directly from the initial map.
2. Predict intermediate latent fields such as "settlement pressure", "trade pressure", or "collapse risk".
3. Rank candidate symbolic rule sets by estimating which ones are promising before full simulation.

The main caution is that an end-to-end network alone will probably learn **where** settlements tend to grow without discovering **why**. That may be good for prediction, but weaker for rule recovery.

The ARC lesson is that neural methods are strongest when they **prune or bias search**, not when they replace the structured hypothesis space entirely.

## Medium transfer

### D. Object-centric representations

Object-centric methods are very relevant, but the "objects" are different.

ARC object-centric solvers usually segment connected components, track shapes, and apply operations to discrete objects. In Astar Island the natural objects are:

- settlement clusters,
- coastline segments,
- fjords and enclosed bays,
- forest patches,
- mountain barriers,
- empty plains corridors,
- and interfaces between settlements and wilderness.

This suggests representing each map at two levels:

1. **Cell level**

   - exact local neighborhood composition,
   - terrain category,
   - coast / inland / mountain-shadow features.

2. **Object level**

   - connected settlement component size,
   - coastal access of a cluster,
   - distance between clusters,
   - whether a frontier cell lies between rival settlement basins,
   - forest-patch size and exposure,
   - narrow-channel / harbor geometry for ports.

ARC-style object-centric decomposition will probably help most for:

- explaining port formation,
- distinguishing isolated settlements from strong clusters,
- and identifying likely conflict frontiers between expanding settlement basins.

### E. Symmetry detection and symmetry-aware augmentation

This transfers in a narrower way than in ARC.

ARC tasks often have exact rotational or reflection symmetries, so D4 or D8 augmentation is powerful. Astar Island maps are not globally symmetric, but the **rules** may still be close to isotropic.

What probably transfers:

- Treat rotated/reflected **local neighborhoods** as equivalent when the geometry is equivalent.
- Canonicalize local templates so that "settlement with forest to the north" and the rotated version "settlement with forest to the east" share parameters.
- Use D4 augmentation when learning local growth/conflict kernels.

What probably does **not** transfer:

- Global whole-map symmetry detection as a primary solving mechanism.

The right abstraction is not "the whole island is symmetric". It is "the local update law is approximately rotation/reflection equivariant except where coastlines or mountains break the symmetry".

## Low transfer

### F. Off-the-shelf exact ARC solvers

Most existing ARC solvers expect:

- 3 to 5 example pairs for one task,
- a deterministic output grid,
- and short programs that solve the task exactly.

Astar Island differs on all three points:

- we have **25 full examples**, not a single few-shot task,
- outputs are **probability distributions**, not exact target grids,
- and the hidden mechanism is a **50-step stochastic simulator**, not a one-shot transformation.

So existing ARC solvers are not likely to work directly without heavy modification.

## 3. How to adapt each requested technique

## DSL-based program synthesis

### Why it fits

- The simulator already has a human-readable phase decomposition: Growth, Conflict, Trade, Winter, Environment.
- The final grid depends on repeated local interactions.
- The state space is small: 6 prediction classes plus a small number of initial terrain codes.

### Best framing

Infer a **probabilistic phase-structured cellular automaton DSL**.

Concretely:

- each phase is a short program,
- each program reads local features and object features,
- each program updates class logits or transition probabilities,
- and the five programs are composed for 50 ticks.

### Recommended search space

- predicates over 1-hop, 2-hop, and 3-hop neighborhoods,
- coast / ocean adjacency,
- connected-component features,
- initial-state memory bits,
- simple arithmetic over counts and thresholds,
- small stochastic kernels rather than deterministic assignments.

### Recommended search algorithm

- evolutionary search,
- beam search over DSL programs,
- or LLM-guided proposal plus deterministic simulation and scoring.

This is the most ARC-like and most promising route if the goal is true reverse engineering.

## Neural pattern matching

### Why it fits

- We have many fully labeled cells across 25 maps.
- The dynamics appear translation-equivariant at local scale.
- The target is probabilistic, which suits neural outputs naturally.

### Best framing

Train a model that predicts either:

- final 6-way probability per cell,
- or hidden pressure fields that parameterize a symbolic simulator.

Good candidates:

- CNN / U-Net with terrain embeddings,
- graph neural net over cells plus component graph,
- ViT-style image-to-image model,
- or test-time-adapted small recurrent model.

### Where it should be used

- as a baseline predictor,
- as a scorer for candidate symbolic programs,
- or as a distillation target after symbolic search discovers approximate rules.

### Where it is weak

- disentangling 50-step causal rules from end-state correlations,
- and giving interpretable automaton rules without extra structure.

## Object-centric representations

### Why it fits

ARC uses objectness because many rules operate on components, not isolated pixels. Astar Island is similar: settlements, coasts, forests, and barriers are structurally meaningful units.

### Useful object extractions

- connected settlement clusters,
- forest components,
- coastline/hinterland partition,
- mountain chains as barriers,
- Voronoi-like settlement influence regions,
- candidate harbor cells on convex / concave coastlines,
- narrow land bridges and choke points for conflict.

### How to use it

- define object-level features in the DSL,
- or let a neural model attend over objects rather than raw cells only.

This is especially valuable because ports and conflict likely depend on geometry larger than a 3x3 patch.

## Symmetry detection

### Why it fits

- Local growth and conflict rules may be rotation/reflection equivariant.
- ARC winners benefited from D8 augmentation; that exact idea can reduce sample complexity here.

### Best use in Astar Island

- canonicalize neighborhood templates under D4 or D8,
- share rule parameters across rotated templates,
- augment training maps by rotations/reflections when testing local-rule hypotheses,
- but preserve orientation-sensitive features like coast access after transformation.

### Caution

Global map symmetry is not the prize here. Local-rule equivariance is.

## 4. Can we frame Astar Island as an ARC task?

## Yes, but only after changing the abstraction level.

### The naive framing

- Input: initial 40x40 grid
- Output: final 40x40 argmax grid

This is ARC-like, but it throws away too much. It ignores uncertainty and the latent multi-step process.

### Better framings

#### A. ARC-style deterministic surrogate tasks

Create separate tasks for:

- initial grid -> final argmax grid,
- initial grid -> settlement-probability heatmap,
- initial grid -> port-probability heatmap,
- initial grid -> entropy map.

This is useful for exploratory modeling and visual reasoning, but only approximates the real problem.

#### B. Local ARC tasks

Treat each cell's neighborhood as the "input grid" and the final class distribution for that cell as the output.

This turns the problem into many small rule-induction tasks and is much closer to a cellular automaton view.

#### C. Program induction over a latent simulator

This is the strongest framing:

- the ARC task is not "paint the final grid";
- the ARC task is "infer the short program that, when iterated 50 times, reproduces the observed probability fields".

That is more like ARC program synthesis than any direct predictor.

## Should we use existing ARC solvers directly?

### Probably not directly

Reasons:

- They expect exact discrete outputs, not distributions.
- They are optimized for one-task few-shot adaptation, not learning one domain across 25 examples.
- Their DSLs are usually about one-shot object transforms, recoloring, cropping, reflection, copying, and counting, not repeated stochastic local updates.

### What is worth reusing

- ARC-style task serialization and visualization
- object extraction utilities
- candidate-program verification loops
- synthetic task generation ideas like `Re-ARC`
- augmentation ideas like D8 symmetry and example reordering
- LLM-guided proposal plus deterministic verifier

### Best hybrid strategy

Use ARC solvers as **infrastructure and inspiration**, not as the final model.

In practice that means:

1. Build an Astar-specific DSL.
2. Build a fast simulator/verifier.
3. Add ARC-style refinement loops.
4. Add a neural prior to rank candidate rule fragments.
5. Use object-centric and symmetry-aware features to reduce the search space.

## 5. Recommended concrete approach for our project

If the goal is to infer the automaton rather than just maximize score, the strongest ARC-inspired pipeline is:

### Phase 1: derive object and local features

- coastal mask
- forest / plains / mountain / ocean masks
- settlement connected components
- local neighbor histograms by radius
- distances to coast, mountain, forest, nearest settlement cluster
- frontier / choke-point / harbor indicators

### Phase 2: define a compact probabilistic CA DSL

- one short rule set per phase
- rules map features to transition logits
- include small stochastic terms
- support D4 canonicalization for local neighborhoods

### Phase 3: refinement-loop search

- initialize rules from hand-coded heuristics and LLM-generated proposals
- simulate 50 ticks on all 25 examples
- score against the probability tensors
- evolve / mutate / prune rules

### Phase 4: neural assistance

- train a CNN or ViT to predict final class probabilities
- use it to rank promising rule programs or to initialize rule parameters
- optionally distill the best symbolic simulator into a neural surrogate for speed

### Phase 5: ablation

Measure how much each ARC-inspired ingredient helps:

- DSL only
- DSL + object features
- DSL + symmetry canonicalization
- DSL + refinement loop
- DSL + neural proposal model
- pure neural baseline

## 6. Practical conclusions

### Most transferable ARC ideas

1. **Program synthesis over a compact DSL**
2. **Generate-verify-refine loops**
3. **Object-centric decomposition**
4. **Symmetry-aware local canonicalization and augmentation**
5. **Neural models as search guides, not sole reasoners**

### Less transferable ARC ideas

1. exact one-shot output-grid solvers
2. one-task few-shot prompting pipelines
3. DSLs built around recoloring/copying/cropping without temporal dynamics

### My recommendation

Treat Astar Island as an **ARC-style latent-program induction problem**, not as a standard supervised image-to-image problem.

If we copy only one thing from ARC, it should be this:

> Search for a short, structured, testable program with a tight verifier, and use neural models only to guide that search.

That matches the actual hidden structure of the simulator much better than pure deep learning.

## Sources

- ARC Prize 2024 competition page: https://arcprize.org/competitions/2024/
- ARC Prize official guide: https://arcprize.org/guide
- ARC Prize 2024 technical report: https://arcprize.org/media/arc-prize-2024-technical-report.pdf
- ARC Prize 2025 results and analysis: https://arcprize.org/blog/arc-prize-2025-results-analysis
- ARC Prize 2025 technical report: https://arxiv.org/abs/2601.10904
- NVARC repository: https://github.com/1ytic/NVARC
- NVIDIA NVARC summary: https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/
- The LLM ARChitect: https://da-fr.github.io/arc-prize-2024/the_architects.pdf
- Combining Induction and Transduction for Abstract Reasoning: https://arxiv.org/abs/2411.02272
- The Surprising Effectiveness of Test-Time Training for Few-Shot Learning: https://arxiv.org/abs/2411.07279
- Towards Efficient Neurally-Guided Program Induction for ARC-AGI: https://arxiv.org/abs/2411.17708
- Searching Latent Program Spaces: https://arxiv.org/abs/2411.08706
- Generalized Planning for the Abstraction and Reasoning Corpus: https://arxiv.org/abs/2401.07426
- arcsolver object-centric tooling: https://agemoai.github.io/arcsolver/index.html
- ARC Is a Vision Problem!: https://arxiv.org/abs/2511.14761
- Vector Symbolic Algebras for ARC: https://arxiv.org/abs/2511.08747
- ARC-AGI-3 page: https://arcprize.org/arc-agi/3/
