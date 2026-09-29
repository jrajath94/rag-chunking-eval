# Synthetic sanity check (real embedder)

Run at 2026-09-28T23:26:34.049169+00:00 (UTC).

Predicted ordering (from TRD Section 8):

hierarchical >= semantic > fixed-500/100 > fixed-500-0

Actual mean recall@5:

- fixed-500-0: 1.000
- fixed-500-100: 1.000
- semantic-0.5: 1.000
- hierarchical-250-1000: 1.000

Result: four-way tie at recall@5 = 1.000. The toy corpus is too easy to discriminate strategies (each document is ~10 sentences, so every strategy retrieves the planted fact sentence). The ordering is neither confirmed nor falsified here; it will be tested for real on the SQuAD slice below. The prediction is not adjusted after the fact.
