### A few classical kernels for discreet timeseries


def LCS_length(seq1, seq2):
    len1, len2 = len(seq1), len(seq2)
    dp = [[0]*(len2+1) for _ in range(len1+1)]
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[len1][len2]

def nLCS(seq1, seq2):
    lcs = LCS_length(seq1, seq2)
    return lcs / ( (len(seq1)*len(seq2))**0.5 )

def spectrum_kernel(seq1, seq2, k=3):
    # Extract k-mers from seq1
    kmers1 = {}
    for i in range(len(seq1) - k + 1):
        kmer = seq1[i:i+k]
        kmers1[kmer] = kmers1.get(kmer, 0) + 1

    # Extract k-mers from seq2 and compute dot product
    val = 0
    for j in range(len(seq2) - k + 1):
        kmer = seq2[j:j+k]
        if kmer in kmers1:
            val += kmers1[kmer]
    return val

def num_mismatches(a, b):
    # Count how many positions differ
    return sum(1 for x, y in zip(a, b) if x != y)

def mismatch_kernel(seq1, seq2, k=3, m=1):
    kmers1 = {}
    for i in range(len(seq1) - k + 1):
        kmer = seq1[i:i+k]
        kmers1[kmer] = kmers1.get(kmer, 0) + 1

    val = 0
    # For each k-mer in seq2, accumulate counts from kmers in seq1 that differ <= m
    for j in range(len(seq2) - k + 1):
        kmer2 = seq2[j:j+k]
        # Check against every kmer1 in seq1 dictionary
        for kmer1, count1 in kmers1.items():
            if num_mismatches(kmer1, kmer2) <= m:
                val += count1
    return val

def gappy_subsequences(seq, k=3, gap=1):
    # Generate all subsequences of length k with a fixed gap between characters
    # For k=3 and gap=1, pattern is indices like (i, i+gap+1, i+2*(gap+1))
    # This is a simplistic approach, assuming a regular spacing pattern.
    subseqs = {}
    length = len(seq)
    step = gap + 1
    # We'll consider subsequences formed by jumping step characters each time
    # until we get a length-k subsequence.
    # For general gappy patterns, you'd consider all subsets, but here we fix a pattern.
    for start in range(length):
        # Try to build a subsequence by jumping step chars
        indices = [start + n*step for n in range(k)]
        if indices[-1] < length:
            subseq = tuple(seq[idx] for idx in indices)
            subseqs[subseq] = subseqs.get(subseq, 0) + 1
    return subseqs

def gappy_kernel(seq1, seq2, k=3, gap=1):
    subseqs1 = gappy_subsequences(seq1, k=k, gap=gap)
    subseqs2 = gappy_subsequences(seq2, k=k, gap=gap)
    # Dot product of frequency vectors
    val = 0
    for s, count1 in subseqs1.items():
        if s in subseqs2:
            val += count1 * subseqs2[s]
    return val

def weighted_degree_kernel(seq1, seq2, k=3, weights=None):
    if weights is None:
        weights = [1.0] * k
    
    length = min(len(seq1), len(seq2))
    val = 0.0
    # Compare prefixes of each position up to length k
    # For each position i, consider substrings seq1[i:i+d], seq2[i:i+d] for d=1..k
    # if still in range.
    for i in range(length):
        for d in range(1, k+1):
            if i + d <= length:
                # Compare substring of length d starting at i
                if seq1[i:i+d] == seq2[i:i+d]:
                    val += weights[d-1]
    return val

def edit_distance(seq1, seq2):
    # Classic dynamic programming approach
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,    # deletion
                dp[i][j-1] + 1,    # insertion
                dp[i-1][j-1] + cost # substitution
            )
    return dp[m][n]

import math

def edit_distance_kernel(seq1, seq2, sigma=1.0):
    dist = edit_distance(seq1, seq2)
    # RBF kernel on edit distance
    return math.exp(-(dist**2) / (2*sigma**2))
