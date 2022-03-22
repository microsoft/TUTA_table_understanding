""" Constants for formula processing and pretraining."""

# formula
NR_AGGR_TO_INDEX = {  # only for nr
    "u+": 0,
    "u-": 1,
    '%': 2,
    '+': 3,
    '-': 4,
    '*': 5,
    '/': 6,
    '^': 7,
    '&': 8,
    '=': 9,
    '<>': 10,
    '>': 11,
    '>=': 12,
    '<': 13,
    '<=': 14,
    'SUM': 15,
    'AVERAGE': 16,
    'MAX': 17,
    'MIN': 18
}
AGGRS = {'SUM', 'AVERAGE', 'MAX', 'MIN'}
UNARY_OPS = {'+', '-'}
BIN_OPS = {'+', '-', '*', '/', '^', '&', '=', '<>', '>', '>=', '<', '<='}
PERCENT = '%'
RANGE_SEP = ':'

# formula prediction
FP_VOCAB = {
    '<START>': 0, '<END>': 1, ':': 2, '<RANGE>': 3, 'C-STR': 4, 'C-NUM': 5, 'C-BOOL': 6,
    '%': 7, '+': 8, '-': 9, '*': 10, '/': 11, '^': 12, '&': 13, '=': 14, '<>': 15, '>': 16, '>=': 17, '<': 18, '<=': 19,
    'SUM': 20, 'IF': 21, 'ROUND': 22, 'VLOOKUP': 23, 'AVERAGE': 24, 'OFFSET': 25, 'ABS': 26, 'EOMONTH': 27,
    'LN': 28, 'MAX': 29, 'ISERROR': 30, 'INDEX': 31, 'MATCH': 32, 'MONTH': 33, 'SQRT': 34, 'AND': 35, 'MIN': 36,
    'EDATE': 37, 'YEAR': 38, 'SUBTOTAL': 39, '<UNKOP>': 40
}
REV_FP_VOCAB = {v: k for k, v in FP_VOCAB.items()}

# formula embedding
FP_START_ID = 110
FP_ENCODE_VOCAB = {f"[{k}]": v + FP_START_ID for k, v in FP_VOCAB.items()}
REV_FP_ENCODE_VOCAB = {v: k for k, v in FP_ENCODE_VOCAB.items()}
