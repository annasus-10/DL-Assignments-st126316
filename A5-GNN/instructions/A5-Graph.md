# A5: Graph Neural Networks — Teaching Guide

---

## Opening: Why Graphs?

**What to say:**

> "Every model we've built assumes data lives on a grid — pixels in a 2D image, tokens in a sequence. But most real-world data doesn't look like that. Social networks, molecules, recommendation systems, knowledge graphs — they're all relational. GNNs are the tool for this."

**Pause and ask:**

> "You want to predict whether a user will like a movie. You have the user's rating history and the movie's genre. What extra information would help — and how would you represent it?"
>
> Answer: Other users who rated the same movies (collaborative filtering), similar movies (co-rating graph). The relationship structure is the information.

---

## Part 1: Graphs and Karate Club (Cells 3–4)

**What to say:**

> "Zachary's Karate Club is the MNIST of GNNs. 34 members of a karate club. A conflict happened — the club split into two factions. The graph shows who socialized with whom. Can a GNN predict which faction each member joined, using only the social links?"

**Key concepts to establish:**

- **Node** = entity (person, movie, atom)
- **Edge** = relationship (friends, co-rated, bonded)
- **Node features** = what we know about each entity
- **Labels** = what we want to predict

**The adjacency matrix A:**

> "A graph with N nodes = an N×N matrix where A[i][j]=1 if there's an edge. Most real graphs are sparse — most entries are 0. We represent this efficiently as an edge list, not a full matrix."

---

## Part 2: The MovieLens Graph (Cells 5–6)

**What to say:**

> "Instead of a toy graph, we'll work with MovieLens-100k — 1,682 movies, connected if the same user rated both. Each node has features: genre one-hot vector + release year. We'll predict each movie's primary genre."

**Why this is interesting:**

> "The graph structure encodes taste similarity — movies that get co-rated tend to be similar. A GNN can use this structure directly. A regular MLP can only use the node features (genre, year) — it ignores who watched what together."

---

## Part 3: GCN (Cells 6–10)

### The core idea — message passing

**What to say:**

> "The idea behind all GNNs is embarrassingly simple: to update a node's representation, aggregate features from its neighbors, then apply a learned transformation."

$$h_v^{(k)} = \sigma\left( W \cdot \text{AGGREGATE}\left(\{ h_u^{(k-1)} : u \in \mathcal{N}(v) \}\right) \right)$$

**Intuition for k layers:**

> "With 1 GCN layer, each node sees its immediate neighbors. With 2 layers, it sees neighbors-of-neighbors — 2-hop neighborhood. With k layers, k-hop. More layers = more context, but too many causes over-smoothing."

### GCN's normalization trick

$$H^{(k+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k)} W^{(k)} \right)$$

**What to say:**

> "Without normalization, nodes with many neighbors dominate — their features get summed 50 times while isolated nodes sum once. $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ divides by the square root of degree on both sides. Every neighbor contributes proportionally."

**The self-loop trick ($\tilde{A} = A + I$):**

> "Add the identity to the adjacency — every node becomes its own neighbor. Without this, a node's current representation is completely replaced by its neighbors' features. The self-loop preserves the node's own information."

---

## Part 4: GAT (Cells 11–16)

### The GCN limitation

**What to say:**

> "GCN treats all neighbors equally. But if you're a drama film, a co-rating with another drama tells you more than a co-rating with an action film. Some neighbors matter more."

**GAT's fix — learned attention:**

$$\alpha_{ij} = \text{softmax}_j\left( \text{LeakyReLU}\left( \mathbf{a}^T [W h_i \| W h_j] \right) \right)$$

> "For each edge (i→j), compute a scalar attention score from the concatenated features of both nodes. Softmax over all neighbors so weights sum to 1. Now aggregation is a *weighted* sum — the model learns which neighbors to trust."

**Multi-head attention:**

> "Run K independent attention mechanisms in parallel, concatenate their outputs. Same idea as Transformer multi-head attention. Different heads can learn to focus on different types of relationships."

**When does GAT beat GCN?**

> "When neighbor importance is heterogeneous — some neighbors are much more informative than others. On homogeneous graphs (all neighbors equally relevant), GAT and GCN perform similarly. GAT is also more robust to noisy edges."

### Connect to Transformers

> "Attention in GAT is the same mechanism as in the Vision Transformer. ViT asks: which *patches* should attend to each other? GAT asks: which *neighbors* should a node attend to? Both learn to weight inputs dynamically instead of using fixed weights."

---

## Bridge: GCN → GAT → GraphSAGE (Cell 17)

### Why scalability matters

**What to say:**

> "GCN and GAT both require the full N×N adjacency matrix in memory. MovieLens has 1,682 nodes — that's fine. But Pinterest has 3 billion nodes. A 3B×3B matrix doesn't fit in any GPU ever built."

**The core problem:**

> "Every GCN/GAT forward pass touches all N nodes simultaneously. You can't batch by nodes — the graph connections couple everything together."

**GraphSAGE's insight:**

> "Instead of aggregating ALL neighbors, randomly sample a fixed number — say k=10. Now each node's computation touches at most k neighbors. Memory per node is constant regardless of graph size. You can train on billion-node graphs with a laptop GPU."

**The concat trick:**

```
GCN:       h_v = σ( Â[v,:] · H · W )
              — weighted sum over ALL neighbors, no self-identity preserved

GraphSAGE: h_v = σ( W · [h_v ∥ mean(h_{N(v)})] )
              — concat self embedding with aggregated neighbors
```

> "The concat keeps the node's own identity separate from what it received from neighbors. This turns out to matter a lot in inductive settings — when the model needs to generalize to nodes it never saw during training."

---

## Part 5: GraphSAGE (Cells 17–19)

### Inductive vs transductive learning

**This is the key conceptual shift from GCN:**

| | Transductive (GCN/GAT) | Inductive (GraphSAGE) |
|---|---|---|
| **Training** | All nodes seen at once | Mini-batch of nodes |
| **Test nodes** | Must be in training graph | Can be new, unseen nodes |
| **Memory** | O(N²) for adjacency | O(k) per node |
| **Use case** | Fixed graph, node labels | Streaming data, new users |

**What to say:**

> "In recommendation systems, new users sign up every day. A transductive model would need to be retrained every time. GraphSAGE learns a *function* — given a node's features and a sample of its neighbors, compute its embedding. New nodes just run the function. No retraining."

### The neighbor sampling trick

**Pause and ask students:**

> "If you randomly sample 10 neighbors instead of all 50, do you lose information?"
>
> Answer: Yes, but you get it back over many training steps. Different epochs sample different subsets. The model learns to generalize from partial neighborhood information — which is exactly what makes it work on new nodes.

### Three-way comparison

After running all three models, the t-SNE comparison reveals:

- **GCN**: Clean clusters, stable. Best if you have the full graph and enough memory.
- **GAT**: Can be highly sensitive to hyperparameters on small training sets. With `dropout=0.6`, `n_heads=8`, `lr=5e-3` on MovieLens (only ~300 labeled training nodes), GAT can collapse to near-random accuracy (observed: 1.2% test acc, val acc stuck near 0 for 150+ epochs) while loss barely moves. This is **not** a sign that attention doesn't work on this graph — it's a sign the defaults (tuned for Cora's much larger label budget) are too aggressive here. Lowering dropout to ~0.3, raising lr to ~1e-2, and reducing heads to 4 (input is only 19-dim) typically fixes it.
- **GraphSAGE**: Comparable or better accuracy than GCN, but the naive per-node Python loop in `GraphSAGELayer.forward` (`for v in range(N): ...`) is **not vectorized** — in practice this makes it the *slowest* of the three per epoch (observed: ~120s total training vs ~9s for GAT and <1s for GCN on the same hardware), the opposite of GraphSAGE's usual "scales better" reputation. Make sure students understand this is an implementation artifact of the teaching code, not a property of the algorithm — real GraphSAGE implementations batch the neighbor sampling.

**Teaching tip:** Have students watch the *loss curve*, not just final accuracy, when something looks off. A flat loss across 100+ epochs (like the GAT failure case) is a stronger signal of a hyperparameter problem than the accuracy number alone — it rules out "the model converged to a bad-but-stable solution" and points at "gradients aren't flowing."

---

## Part 6: Recommendation System (Cells 21–24)

### The bipartite graph idea

**What to say:**

> "So far we predicted node labels (genre classification). GNNs can also predict *edges* — link prediction. In recommendation: does user U like movie M? That's predicting whether an edge should exist in the user-movie bipartite graph."

```
Users        Movies
  U1 ────── M1 (rated)
  U1 ────── M3 (rated)
  U2 ────── M1 (rated)
  U2 ──?──  M3 (not yet rated — should we recommend?)
```

**The GNN approach:**

> "Embed users and movies into the same vector space using GCN on the bipartite graph. Users who rated similar movies get similar embeddings. Movies rated by similar users get similar embeddings. Predict rating = dot product of user and movie embeddings."

**Why GNN beats matrix factorization:**

> "Matrix factorization only uses the rating matrix. GNN can additionally use node features — movie genre, user demographics. And the graph structure itself propagates information: if U1 and U2 have similar taste, M3's embedding is influenced by U1's ratings even though U1 never rated M3."

---

## Exercise Walkthrough Tips

**Ex1 (Over-smoothing):** With 1 layer, nodes only see 1-hop neighbors. With 5+ layers, all nodes average over the whole graph — embeddings converge to the same value. Students should see accuracy *drop* after 3–4 layers. Plot both test accuracy and average cosine similarity vs depth. When cosine similarity approaches 1.0, over-smoothing is severe. Mechanical explanation: each layer applies $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$, which is a low-pass filter. Stack enough layers and all frequencies are smoothed out.

**Ex2 (Three-way comparison):** Students often assume GAT wins because it's more complex. In practice GAT is the model most likely to need debugging: with the default hyperparameters in this notebook it can collapse to near-random accuracy on MovieLens (val accuracy stuck near 0, loss barely decreasing over 200 epochs) because dropout=0.6 and 8 heads are tuned for Cora's larger labeled set, not MovieLens's ~300 training nodes. If a student reports GAT accuracy near 1–2%, walk them toward lowering dropout (~0.3) and learning rate up (~1e-2) rather than assuming attention "doesn't work" on this graph. Separately, GraphSAGE's per-node Python loop is unvectorized, so it is usually the *slowest* of the three per epoch here — not the fastest, despite the algorithm's reputation for scaling. For the attention visualization (once GAT is actually training): expect top-attended neighbors to share genre labels ~60–70% of the time (not 100% — some edges are cross-genre by design).

**Ex3 (MLP baseline):** This is the most important sanity check in the lab. If GCN can't beat a 2-layer MLP on node features alone, the graph structure isn't helping. On MovieLens, GCN typically beats MLP by 5–10% because genre features alone are weak (18 binary flags, many movies have the same genre) while co-rating provides rich collaborative signal. Key discussion: "relational information is irreducible — you can't compress the graph into the node features."

**Ex4 (LightGCN):** Students expect more parameters = better. LightGCN removes W and σ entirely — only trainable parameters are the initial user/item embeddings. It outperforms RecGCN on recommendation because removing W avoids feature transformation that is unhelpful when input features are already learned embeddings (not raw features). The over-smoothing question: LightGCN uses a *layer-averaged* final embedding (sum across all K layers weighted equally). This prevents the final representation from being dominated by the deepest layer — over-smoothing is less severe than GCN.

---

## The Big Picture

```
2009  GNN (Scarselli)    "First framework: aggregate neighbor messages iteratively."
        ↓
2017  GCN (Kipf)         "Spectral convolution → simple matrix form. Symmetric normalization."
        ↓
2017  GraphSAGE          "Sample neighbors, not full graph. Scales to millions of nodes."
        ↓
2018  GAT (Veličković)   "Learned attention weights per edge. Not all neighbors equal."
        ↓
2020+ Graph Transformers "Full self-attention over all nodes + structural encodings."
```

**Note on ordering:** GCN and GraphSAGE are both 2017 papers. GCN came first (Feb) and is simpler to explain. GraphSAGE (June) was explicitly motivated by GCN's scalability problem.

**Closing question:**

> "You're building a drug discovery system. Atoms are nodes, chemical bonds are edges, node features are atom type and charge. You want to predict if a molecule is toxic. How many GCN layers would you use, and why?"
>
> Answer: 3–4 layers. A molecule's toxicity depends on functional groups (local structure, 2–3 hops) and global shape (4–5 hops). Too many layers → over-smoothing, all atoms look the same. Sweet spot: enough to capture the relevant subgraph structures without losing local identity.

**Follow-up:**

> "Now suppose new molecules arrive every hour and you need to embed them immediately without retraining. Which architecture — GCN or GraphSAGE — would you use? Why?"
>
> Answer: GraphSAGE. It learns an *inductive* function — given a new molecule's atom features and bond structure, compute its embedding on the fly. GCN is transductive: it embeds the full training graph and can't generalize to new nodes without retraining.
