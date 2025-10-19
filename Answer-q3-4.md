# Q3–Q4 Write-Up

## Q3

### (a)

Base on the recognizer, each grammar line is read into a `Rule` whose weight is the negative log-probability, so we can accumulate costs by addition. An Earley item still records `(rule, dot_position, start_position)`, but the column agenda now maintains dictionaries for the current best weight and the backpointer of every item. Predict enqueues a new item with the rule’s weight and a sentinel backpointer that marks the start of the derivation. Scan propagates the parent weight unchanged because terminals have no additional cost, while attach merges the parent prefix weight with the completed child’s lowest score before advancing the dot. Whenever the chart finishes, we sweep the last column for complete ROOT items that span the whole sentence, the backpointer chain of the lightest one is then walked recursively to print the minimum weight tree.

### (b)

`Agenda.push_or_move` keeps a single entry per rule. When an item appears for the first time it is stored, indexed under its pending next symbol, and associated with its weight/backpointer pair. If later processing discovers the same dotted rule with a smaller weight, this method overwrites the weight and backpointer, and if the item had already been popped, un-pops it by swapping it just before `_next`, ensuring the improved item will be processed again. In this way the chart performs exact lowest score search without producing duplicate queue entries. Since there are only `n+1` columns and each item is uniquely determined by its start position, dot position, and grammar rule, the total number of stored items is $O(n^2)$, matching the usual Earley space bound. Every push (including the duplicate check) is $O(1)$: it uses a single hash lookup in `_index`, an optional append to `_items`, and a constant number of dictionary updates when registering the item in `_waiting`. That constant-time agenda maintenance is what lets the predict/scan/attach schedule stay within the standard $O(n^3)$runtime—any slower push would blow up that bound because each completed item can trigger $O(n)$ further operations in $O(n)$columns.

### Reprocessing (extra credit; reading section B.2)

I followed B.2 "move" reprocessing: when `push_or_move` finds a lower-weight duplicate, it overwrites the item’s stored weight/backpointer and swaps the item back into the unpopped region of the column in O(1) so it will be processed again. Since all rule weights are −log probabilities (≥0) and scans add 0, weights are monotone, hence every improvement strictly lowers the value, so this cannot loop.

Let V be the number of distinct items, which is $O(n^2)$ for a fixed grammar, and let E be the number of attach “uses,” which is $\Theta(n^3)$. Each time Z improves, we redo its $O(\text{outdegree}(Z))$ attachments. In highly ambiguous grammars, Z could theoretically improve many times (once per distinct derivation path to Z). More precisely, the total runtime is

$$\Theta(n^3)\; +\; \sum_Z (\#\text{improvements of }Z)\cdot \text{outdegree}(Z).$$

To avoid reprocessing, we could use a Global best‑first agenda (Dijkstra). Replace the per‑column FIFO with one global min‑heap keyed by item weight. With non‑negative edge costs (rule ≥0, attach adds a completed child’s cost ≥0, scan 0), once an item pops from the heap, its weight is final and can never be improved later. Hence no reprocessing is needed. The complexity is $O((V+E) \log V) = O(n^3 \log n)$ with a binary heap.

## Q4

I profiled various speedups on `wallstreet.gr`/`wallstreet.sen`. The main hotspots were repeated predictions and huge numbers of dotted-rule states. I addressed those issues in stages and the final submitted `parse2.py` combines the most effective optimizations:

- **Prediction memoization (E.1).** The first attempt maintains a column-level cache so each `(nonterminal, position)` pair is expanded only once. This prevents every customer from re-reading the same rule list. Since later I truned to Trie for grammars so this is not used anymore in the final submission.
- **Vocabulary specialization (E.2).** For the sentence being parsed I collect all token ids and restrict predictions to nonterminals that can reach at least one of those terminals, plus the start symbol. The grammar tracks reachable terminals per nonterminal while it is building its tries. Items whose outgoing symbols fall outside this allow-set are never scheduled, which prunes large sections of the chart.
- **Trie-based Earley items (E.4).** Each left-hand side owns a trie that merges all right-hand-side prefixes and stores the rule weight at terminal nodes. An item is now `(lhs id, trie state, start_position)`, so a single state represents every dotted rule that shares the same prefix. When an item pops, I enumerate all outgoing nonterminal for prediction, follow any matching terminal to scan, and when a node is final I add the cached rule weight while attaching to waiting parents. The agenda indexes each item under all of its future nonterminals, which keeps attach lookups constant time.
- **Integerized symbols.** I reused the `Integerizer` from previous homeworks to map every terminal and nonterminal to an id as soon as the grammar loads. The sentence tokens are cached as ids, trie edges are stored as integer-labelled transitions, and the agenda’s waiting dictionary is keyed by integer nonterminals. This removes the repeated hashing of Python strings inside all hot loops.

Together these changes shrink both the number of states we explore and the per-state processing cost, which is reflected in the expriment results below.

### Experiment results

Wall Street parsing times (Using PyPy on M3 Max):

| parser used | time taken (s) |
|-------------|----------------|
| baseline probabilistic parser (`parse.py`) | 902.43 |
| + E.1 batch duplicate check | 279.82 |
| + E.1 and E.2 vocabulary specialization | 265.60 |
| + E.2 and E.4 trie items | 75.69 |
| + E.2, E.4, and integerization (final `parse2.py`) | 71.97 |

The final submission is over 12× faster than the unoptimized parser on this benchmark while still returning identical best parses.
