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