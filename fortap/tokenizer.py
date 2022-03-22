#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize tables (the reader output) to generate model inputs.
"""

import random
import collections
import unicodedata

from utils import *
from formula_utils import *
from constants import (
    NR_AGGR_TO_INDEX,
    FP_VOCAB,
    REV_FP_VOCAB,
    FP_ENCODE_VOCAB,
    REV_FP_ENCODE_VOCAB,
    FP_START_ID
)


# %% Special Token Identifications
# additionally defined token ids
VAL_ID = 1
EMP_ID = 2  # placeholder for emtpy cells
# adopt from bert
PAD_ID = 0
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
FORMULA_ID = 104

# corresponding token-strings
VAL_TOKEN = "[VAL]"
EMP_TOKEN = "[EMP]"

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
FORMULA_TOKEN = "[FORMULA]"

# constant
NOTUSE_TAG = 10000
SR_NEG_TAG = 0
SR_POS_TAG = 1
SR_FORMULA_TAG = 2
SR_CONTEXT_NORMAL_TAG = 0
SR_CONTEXT_ANCHOR_HEADER_TAG = 1
SR_CONTEXT_REF_HEADER_TAG = 2
SR_CONTEXT_REF_DATA_CELL_TAG = 3
SR_CONTEXT_FORMULA_HEADER_TAG = 4
SR_CONTEXT_FORMULA_CELL_TAG = 5
DEFAULT_SR_LABEL = -1
DEFAULT_SR_CONTEXT_LABEL = -1
# NR_FORMULA_TAG = -1  # -1 is not valid for scatter
DEFAULT_NR_LABEL = len(NR_AGGR_TO_INDEX)
FP_PAD_TAG = len(FP_VOCAB)
DEFAULT_RANGE_LABEL = 0  # the first tok is [CLS], which mustn't be `range`
DEFAULT_OP_MLM_LABEL = -1
DEFAULT_RANGE_MLM_LABEL = 0


# %% Cleansing functions
def whitespace_split(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: {}".format(type(text)))


def load_vocab(vocab_path):
    vocab = collections.OrderedDict()
    with open(vocab_path, "r", encoding='utf-8') as reader:
        for index, line in enumerate(reader):
            token = convert_to_unicode(line.strip())
            if not token:
                continue
            vocab[token] = index
    print("Successfully Loaded Vocabulary from {}".format(vocab_path))
    return vocab


# %% Basic
class BasicTokenizer(object):
    """ Basic: separate words at whitespaces, punctuations, and numbers"""

    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.digit_set = set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        self.numeric_mark = set([',', '.'])

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        org_text_list, org_type_list = self._group_digit_chars(text)  # cut numbers apart

        text_list, type_list = [], []
        for stext, stype in zip(org_text_list, org_type_list):
            if stype:  # digit span (with , and .)
                stext = self.remove_comma_in_number(stext)
                slist, stype_list = self.split_by_dot(stext)
                text_list.extend(slist)
                type_list.extend(stype_list)
            else:  # text span
                stext_list = whitespace_split(stext)
                for item in stext_list:
                    if self.do_lower_case:
                        item = item.lower()
                        item = self._run_strip_accents(item)  # str removed with accent tokens
                    item = self._run_split_on_punc(item)
                    text_list.extend([ii.strip() for ii in item])
                    type_list.extend([stype for _ in item])

        stripped_text, stripped_type = [], []
        for stext, stype in zip(text_list, type_list):
            if stext.strip():
                stripped_text.append(stext.strip())
                stripped_type.append(stype)
        stripped_text = whitespace_split(" ".join(stripped_text))  # not necessary
        assert len(stripped_text) == len(stripped_type), "sizes of text and type don't match."
        return stripped_text, stripped_type  # list of texts without whitespaces

    def remove_comma_in_number(self, text):
        """ convert '3,000' to '3000' """
        chunks = text.split(',')
        return "".join([c for c in chunks if c])

    def split_by_dot(self, digit_text):
        """ 
        Handle digits in numbers, e.g. '3.00', '3' or '192.0.0.0'.
        treat as respective numbers if >=2 dots, value strings remain otherwise.
        """
        text_list, type_list = [], []
        parts = [p for p in digit_text.strip().split('.') if p]
        if len(parts) > 2:  # not a value
            for part in parts:
                text_list.append(part)
                type_list.append(True)
                text_list.append('.')
                type_list.append(False)
            text_list = text_list[: -1]
            type_list = type_list[: -1]
        elif len(parts) > 0:  # integer or decimal
            text_list.append(digit_text)
            type_list.append(True)
        return text_list, type_list

    def _group_digit_chars(self, text):
        """split numbers apart from surrounding texts """
        output, type_list = [[]], []
        digit_flag = False
        if text and (text[0] in self.digit_set):
            digit_flag = True

        for char in text:
            if char in self.numeric_mark:
                output[-1].append(char)
            elif char in self.digit_set:  # current item is digit
                if digit_flag is True:  # previous item also a digit
                    output[-1].append(char)
                else:  # alter from text to digit
                    type_list.append(digit_flag)  # type mark of previous span
                    digit_flag = True
                    output.append([char])
            else:  # current item is text
                if digit_flag is False:  # previous item also text
                    output[-1].append(char)
                else:  # alter from digit to text
                    type_list.append(digit_flag)
                    digit_flag = False
                    output.append([char])
        text_list = ["".join(span) for span in output]
        type_list.append(digit_flag)
        return text_list, type_list

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True
        return False

    def _clean_text(self, text):
        cleaned_text = ""
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                cleaned_text += " "
            else:
                cleaned_text += char
        return cleaned_text


# %% Word piecing
class WordpieceTokenizer(object):
    """Cut words into token pieces """

    def __init__(self,
                 vocab, unk_token="[UNK]", val_token="[VAL]",
                 mag_size=10, pre_size=10, top_digit=10, low_digit=10,
                 max_cell_len=8, max_token_chars=64
                 ):
        self.vocab = vocab
        self.unk_token = unk_token
        self.val_token = val_token
        self.mag_size = mag_size
        self.pre_size = pre_size
        self.top_digit = top_digit
        self.low_digit = low_digit
        self.default_num = (
            self.mag_size + 1,
            self.pre_size + 1,
            self.top_digit + 1,
            self.low_digit + 1
        )
        self.max_cell_len = max_cell_len
        self.max_token_chars = max_token_chars

    def tokenize(self, word_text, word_type):
        """ Cut a single word into tokens using the vocabulary """
        word_text = convert_to_unicode(word_text)
        if word_type is True:  # digit span
            return self.tokenize_digit(word_text)

        output_tokens = []
        for text in whitespace_split(word_text):
            chars = list(text)
            if len(chars) > self.max_token_chars:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens, [self.default_num for _ in output_tokens]

    def tokenize_digit(self, digit_text):
        """ 
        Tokenize numeric strings.
        Default settings:
            magnitude: +inf (= magnitude_size), nan (= magnitude_size + 1)
            precision: +inf (= precision_size), nan (= precision_size + 1)
        """
        parts = [p for p in digit_text.strip().split('.') if p]
        token, magnitude, precision = self.val_token, 0, 0
        if len(parts) == 1:  # integer
            if digit_text in self.vocab:
                token = digit_text
            magnitude = len(digit_text)
        if len(parts) == 2:  # decimal
            magnitude = len(parts[0])
            precision = len(parts[1])
        top, low = int(parts[0][0]), int(parts[-1][-1])

        # post-process
        magnitude = min(magnitude, self.mag_size)
        precision = min(precision, self.pre_size)
        top = min(max(0, top), self.top_digit)
        low = min(max(0, low), self.low_digit)
        return [token], [(magnitude, precision, top, low)]


# %% Leveraged for (matrix) tables
class TableTokenizer(object):
    """Basic definitions, string tokenization, repository building. """

    def __init__(self, args, do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.vocab = load_vocab(args.vocab_path)  # get id from token (str)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}  # get token str from id
        self.punc_ids = set(
            [index - 1 for index in range(1000, 1004)] +
            [index - 1 for index in range(1025, 1037)] +
            [index - 1 for index in range(1064, 1996)] +
            [index - 1 for index in range(29613, 30522)]
        )
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab,
            mag_size=args.magnitude_size,
            pre_size=args.precision_size,
            top_digit=args.top_digit_size,
            low_digit=args.low_digit_size
        )

        self.row_size = args.row_size
        self.column_size = args.column_size
        self.tree_depth = args.tree_depth
        self.node_degree = args.node_degree
        self.max_cell_length = args.max_cell_length
        self.max_seq_len = args.max_seq_len
        self.text_threshold = args.text_threshold
        self.value_threshold = args.value_threshold
        self.clc_rate = args.clc_rate
        self.wcm_rate = args.wcm_rate  # whole-cell-mask rate

        self.num_format_feature = args.num_format_feature
        self.default_format = [0.25, 0.25, 0., 0., 0., 0., 0., 0., 0., 1., 1.]
        self.format_range = [4., 4., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        self.default_mlm_label = -1
        self.default_clc_label = 0
        self.total_node = sum(self.node_degree)
        self.default_tree_position = [-1 for _ in self.node_degree]

        # load pre-collected context tuples and cell strings in repository
        self.context_repo = []
        self.load_repository(self.context_repo, args.context_repo_path)

        self.cellstr_repo = {keylen: [] for keylen in range(2, args.max_cell_length + 1)}
        self.build_cellstr_dict(self.cellstr_repo, args.cellstr_repo_path)

    def load_repository(self, repository, repo_path):
        if repo_path is None:
            return
        with open(repo_path, "r", encoding='utf-8') as fr:
            lines = fr.readlines()
        repository.extend([line.strip() for line in lines])
        print("tokenizer collected {} context pieces in repository.".format(len(self.context_repo)))

    def build_cellstr_dict(self, repository, repo_path):
        if repo_path is None:
            return
        with open(repo_path, "r", encoding='utf-8') as fr:
            lines = fr.readlines()
        print("#Num of Lines: ", len(lines))
        for line in lines:
            cellstr = line.strip()
            tokens, _ = self.tokenize_text(cellstr, add_separate=False)
            key_len = len(tokens)
            if key_len < 2:
                continue
            for small_key_len in range(len(tokens), 1, -1):
                if len(repository[small_key_len]) < len(repository[key_len]):
                    key_len = small_key_len
            offset = random.randint(0, len(tokens) - key_len)
            repository[key_len].append(tokens[0 + offset: key_len + offset])

        # sampling, then print repository message
        counts = []
        for key_len in repository.keys():
            m = min(len(repository[key_len]), 500)
            counts.append(m)
            repository[key_len] = random.sample(repository[key_len], m)
        msg = ", ".join([str(l) for l in counts])
        print("Replace Repository: {}".format(msg))

    def tokenize_string_matrix(self, string_matrix, add_separate, max_cell_len=8):
        """ Return cell-wise text/number ids as lists. """
        token_matrix, number_matrix = [], []
        for string_row in string_matrix:
            token_matrix.append([])
            number_matrix.append([])
            for cell_string in string_row:
                cell_token, cell_number = self.tokenize_text(cell_string, add_separate, max_cell_len)
                token_matrix[-1].append(cell_token)
                number_matrix[-1].append(cell_number)
        return token_matrix, number_matrix

    def tokenize_text(self, cell_string, add_separate=True, max_cell_len=8):
        """ 
        Tokenize cell strings (as text) upto max_cell_len. 
        cell_number = (magnitude, precision, highest, lowest), or default value
        """
        cell_token, cell_number = [], []
        if add_separate == True:
            cell_token.append("[SEP]")
            cell_number.append(self.wordpiece_tokenizer.default_num)
        text_list, type_list = self.basic_tokenizer.tokenize(cell_string)
        for word_text, word_type in zip(text_list, type_list):
            token_list, num_list = self.wordpiece_tokenizer.tokenize(word_text, word_type)
            cell_token.extend(token_list)
            cell_number.extend(num_list)
        cell_token = self.convert_tokens_to_ids(cell_token)
        assert len(cell_token) == len(cell_number), "Token number doesn't match Magnitudes."
        if len(cell_token) == 1 and cell_token[0] == SEP_ID:
            cell_token.append(EMP_ID)
            cell_number.append(self.wordpiece_tokenizer.default_num)
        cell_token = cell_token[: max_cell_len]
        cell_number = cell_number[: max_cell_len]
        if add_separate == True:
            assert len(cell_token) > 1, "Token list too short: {} -> {}".format(cell_string, cell_token)
        return cell_token, cell_number

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, UNK_ID))
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for token_id in ids:
            token = self.inv_vocab.get(token_id, '[UNK]')
            tokens.append(token)
        return tokens


class TutaTokenizer(TableTokenizer):
    def check_valid(self, tokens, numbers, in_header):
        """
        Check if a cell is valid enough as table input
            remove empty cell, in both header and data regions
            for all-number cell, keep in header (do not mask), sampling from data
            for half-text&number cell, keep in header (can mask), sampling from data
            for text cell, keep in header (can mask), sampling from data (resp.)
        """
        if EMP_ID in tokens:
            return 0
        if self.wordpiece_tokenizer.default_num not in set(numbers[1:]):  # pure numeric
            if (in_header == True) or (random.random() < self.value_threshold):
                return 1
        elif set([self.wordpiece_tokenizer.default_num]) == set(numbers[1:]):  # pure text
            if (in_header == True) or (random.random() < self.text_threshold):
                return 2
        else:  # halfway
            if (in_header == True) or (random.random() < self.value_threshold):
                return 2
        return 0

    def sampling(self,
                 token_matrix, number_matrix, header_info,
                 max_disturb_num=20, disturb_prob=0.2, clc_rate=0.3
                 ):
        """ Mark each cell: '0' as dumped, '1' as input, '2' as blanked, '3' as masked. """
        header_rows, header_columns = header_info
        sampling_mask = [[1 for _ in token_matrix[0]] for _ in token_matrix]
        divide_prob = random.random()
        divide_prob = 0.4 + divide_prob / 5

        mask = []
        valid_cell_count = 0
        blank_buckets = [set() for _ in range(header_rows + header_columns + 2)]
        # coarse marking
        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                in_header = (irow < header_rows) or (icol < header_columns)
                cell_valid = self.check_valid(tokens, number_matrix[irow][icol], in_header)
                valid_cell_count += cell_valid
                if cell_valid == 0:
                    sampling_mask[irow][icol] = 0
                elif cell_valid > 1:
                    prob = random.random()
                    if prob < (disturb_prob * divide_prob):  # mask
                        mask.append((irow, icol))
                    else:  # blank or none
                        if irow < header_rows:
                            if icol < header_columns:
                                blank_buckets[0].add((irow, icol))
                            else:
                                blank_buckets[1 + irow].add((irow, icol))
                        elif icol < header_columns:
                            blank_buckets[header_rows + 1 + icol].add((irow, icol))
                        else:
                            blank_buckets[-1].add((irow, icol))

        max_disturb_num = min(max_disturb_num, int(disturb_prob * valid_cell_count))
        # refine mask marks
        mask_num = min(len(mask), int(max_disturb_num * divide_prob))
        mask = random.sample(mask, mask_num)
        for irow, icol in mask:
            sampling_mask[irow][icol] = 3

        # refine blank marks
        blank = []
        for bucket in blank_buckets:
            if len(bucket) == 0:
                continue
            if (len(bucket) < 6) and (random.random() < clc_rate):
                blank.extend(random.sample(bucket, 1))
            else:
                bucket_blank_num = min(max(1, int(len(bucket) * clc_rate)), 3)
                bucket_blank = random.sample(bucket, bucket_blank_num)
                blank.extend(bucket_blank)
        blank_num = min(len(blank), max_disturb_num - mask_num)
        if blank_num > 1:
            blank = random.sample(blank, blank_num)
            for irow, icol in blank:
                sampling_mask[irow][icol] = 2
        return sampling_mask

    def init_table_seq(self, root_context):
        """Initialize table sequence with CLS_ID at head, add context if provided. """
        context_tokens, context_number = self.tokenize_text(cell_string=root_context, add_separate=False,
                                                            max_cell_len=8)
        if len(context_tokens) > 0:  # do context mlm if available
            gold_labels, masked_tokens, _ = self.easy_mask(context_tokens, context_number)
            token_list = [[CLS_ID] + masked_tokens]
            mlm_label = [[self.default_mlm_label] + gold_labels]
        else:
            token_list = [[CLS_ID]]
            mlm_label = [[self.default_mlm_label]]
        num_list = [[self.wordpiece_tokenizer.default_num] + context_number]
        pos_list = [(self.row_size, self.column_size, self.default_tree_position, self.default_tree_position)]
        format_list = [self.default_format]
        indicator = [[-1] + [-2 for _ in context_tokens]]
        clc_label = [[self.default_clc_label for _ in token_list[0]]]
        cell_num = 1
        seq_len = len(token_list[0])
        return token_list, num_list, pos_list, format_list, indicator, mlm_label, clc_label, cell_num, seq_len

    def get_text_choices(self, context_truths, length_left, max_pair_num=3):
        choice_label_pairs = []
        context_length, num = 0, 0
        for truth_text in context_truths:
            truth_token, truth_num = self.tokenize_text(truth_text, True, 20)
            disturb_text = self.context_repo.pop()  # pair each truth with a disturb
            disturb_token, disturb_num = self.tokenize_text(disturb_text)
            context_length += len(truth_token) + len(disturb_token)
            if (context_length > length_left) or (num > max_pair_num):
                self.context_repo.append(disturb_text)  # push the disturb back if not used
                break
            num += 1
            choice_label_pairs.append(((truth_token, truth_num), 1))
            choice_label_pairs.append(((disturb_token, disturb_num), 0))

        new_repo = random.sample(context_truths, num)
        self.context_repo.extend(new_repo)
        random.shuffle(choice_label_pairs)
        return choice_label_pairs

    def objective_preprocess(self,
                             sampling_matrix, token_matrix, number_matrix, position_lists,
                             format_matrix=None, context=None, add_sep=True
                             ):
        top_pos_list, left_pos_list = position_lists
        row_number, column_number = len(token_matrix), len(token_matrix[0])
        # rewrite a fake format_matrix (all default values) for wiki & wdc tables
        if format_matrix is None:
            format_matrix = [[[0. for _ in range(self.num_format_feature)] for _ in range(column_number)] for _ in
                             range(row_number)]

        # get all context chunks
        context_truths = self.get_context_truth(context)
        root_context = ""
        if len(context_truths) > 0:
            root_context = context_truths[0]
            context_truths = context_truths[1:]

        token_list, num_list, pos_list, format_list, \
        indicator, mlm_label, clc_label, cell_num, seq_len = self.init_table_seq(root_context=root_context)
        paste_token_list, paste_num_list, paste_clc_label = [], [], []

        for irow, sampling_row in enumerate(sampling_matrix):
            for icol, mark in enumerate(sampling_row):
                if mark == 0:  # dumped
                    continue
                cell_num += 1
                tokens = token_matrix[irow][icol]
                number = number_matrix[irow][icol]
                cell_len = len(tokens)
                if mark == 1:  # normal
                    token_list.append(tokens)
                    num_list.append(number)
                    mlm_label.append([self.default_mlm_label for _ in tokens])
                    clc_label.append([self.default_clc_label for _ in tokens])
                elif mark == 2:  # blank
                    paste_token_list.append(tokens)
                    paste_num_list.append(number)
                    if add_sep == True:
                        token_list.append([SEP_ID, EMP_ID])
                        num_list.append([self.wordpiece_tokenizer.default_num] * 2)
                        mlm_label.append([self.default_mlm_label, self.default_mlm_label])
                        clc_label.append([-seq_len, -seq_len - 1])
                        assert cell_len > 1
                        real_clc_label = [seq_len, seq_len + 1] + [0] * (cell_len - 2)  # [SEP] people's repub of china
                        paste_clc_label.append(real_clc_label)
                        cell_len = 2
                    else:
                        token_list.append([EMP_ID])
                        num_list.append([self.wordpiece_tokenizer.default_num])
                        mlm_label.append([self.default_mlm_label])
                        clc_label.append([-seq_len])
                        assert cell_len > 1
                        real_clc_label = [seq_len] + [0] * (cell_len - 1)
                        paste_clc_label.append(real_clc_label)
                        cell_len = 1
                elif mark == 3:  # mask
                    if add_sep == True:
                        if random.random() < self.wcm_rate:
                            gold, fake, sep_label = self.whole_mask(tokens[1:], number[1:])
                        else:
                            gold, fake, sep_label = self.easy_mask(tokens[1:], number[1:])
                        token_list.append([SEP_ID] + fake)
                        mlm_label.append([sep_label] + gold)
                    else:
                        gold, fake = self.easy_mask(tokens, number)
                        token_list.append(fake)
                        mlm_label.append(gold)
                    num_list.append(number)
                    clc_label.append([self.default_clc_label for _ in tokens])

                # add corresponding format vector
                format_vector = []
                for ivec, vec in enumerate(format_matrix[irow][icol]):
                    format_vector.append(min(vec, self.format_range[ivec]) / self.format_range[ivec])
                format_list.append(format_vector)

                icell = irow * column_number + icol
                # check if has null position
                if (max(top_pos_list[icell]) == -1) or (max(left_pos_list[icell]) == -1):
                    return None
                pos_list.append((irow, icol, top_pos_list[icell], left_pos_list[icell]))
                indicator.append([cell_num * 2 for _ in range(cell_len)])
                if add_sep:
                    assert (cell_len > 1) or (token_list[-1] == [CLS_ID]), "Mini cell: {}".format(token_list[-1])
                    indicator[-1][0] -= 1
                seq_len += cell_len

        paste_position = (self.row_size, self.column_size, [-1] * self.tree_depth, [-1] * self.tree_depth)
        for tokens in paste_token_list:
            cell_num += 1
            seq_len += len(tokens)
            # ramdom positions for pasted cells to enlarge the attention distance with other cells
            paste_position = (self.row_size, self.column_size, [random.randint(0, 31)] * self.tree_depth,
                              [random.randint(0, 31)] * self.tree_depth)

            pos_list.append(paste_position)
            format_list.append(self.default_format)
            mlm_label.append([self.default_mlm_label for _ in tokens])
            paste_ind = [cell_num * 2] * len(tokens)
            if add_sep:
                paste_ind[0] -= 1
            indicator.append([-dd for dd in paste_ind])
        token_list.extend(paste_token_list)
        num_list.extend(paste_num_list)
        clc_label.extend(paste_clc_label)

        # add table-level context choices
        tcr_label = []
        for clc in clc_label:
            tcr_label.append([-1 for _ in clc])

        context_choice_label_pairs = self.get_text_choices(context_truths, self.max_seq_len - seq_len)
        # adjust number of choices based on current sequence length
        for (token, number), label in context_choice_label_pairs:
            cell_num += 1
            token_list.append(token)
            num_list.append(number)
            mlm_label.append([self.default_mlm_label for _ in token])
            paste_position = (self.row_size, self.column_size, [random.randint(0, 31)] * self.tree_depth,
                              [random.randint(0, 31)] * self.tree_depth)
            pos_list.append(paste_position)
            format_list.append(self.default_format)
            indicator.append([-cell_num * 2 for _ in token])
            if add_sep:
                indicator[-1][0] += 1
            clc_label.append([self.default_clc_label for _ in token])
            tcr_label.append([-1 for _ in token])
            tcr_label[-1][1] = label
            if add_sep:
                tcr_label[-1][0] = label
        return token_list, num_list, pos_list, format_list, indicator, mlm_label, clc_label, tcr_label

    def get_replace_token(self, token_id):
        prob = random.random()
        if prob < 0.8:
            return MASK_ID
        elif prob < 0.9:
            return random.randint(1996, 29611)
        else:
            return token_id

    def get_mlm_index(self, cell_ids, cell_num):
        """ Select one token in list, prefer text over punctuations over numbers. """
        nonum_indexes, text_indexes = [], []
        for ii, (ids, num) in enumerate(zip(cell_ids, cell_num)):
            if self.wordpiece_tokenizer.default_num == num:
                nonum_indexes.append(ii)
                if (ids not in self.punc_ids):
                    text_indexes.append(ii)
        if len(text_indexes) > 0:
            index = random.sample(text_indexes, 1)[0]
        elif len(nonum_indexes) > 0:
            index = random.sample(nonum_indexes, 1)[0]
        else:
            index = random.randint(0, len(cell_ids) - 1)
        return index

    def easy_mask(self, cell_ids, cell_num):
        """
        Mask only one token in the given token list.
        get a random index, mark golden truth, and build fake token list

        inputs: token_ids and numerical_feats of a cell
        return: gold labels, fake token list, [SEP] label
        """
        index = self.get_mlm_index(cell_ids, cell_num)
        gold = [-1 for _ in cell_ids]
        gold[index] = cell_ids[index]
        fake = cell_ids[: index] + [self.get_replace_token(cell_ids[index])] + cell_ids[index + 1:]
        sep_label = -1  # cell_ids[index]
        return gold, fake, sep_label

    def get_mlm_index_whole(self, cell_ids, cell_num):
        """ Record all viable tokens in list, prefer text over punctuations over numbers. """
        nonum_indexes, text_indexes = [], []
        indexes = []
        for ii, (ids, num) in enumerate(zip(cell_ids, cell_num)):
            if self.wordpiece_tokenizer.default_num == num:
                nonum_indexes.append(ii)
                if (ids not in self.punc_ids):
                    text_indexes.append(ii)
        if len(text_indexes) > 0:
            indexes = text_indexes
        elif len(nonum_indexes) > 0:
            indexes.append(random.sample(nonum_indexes, 1)[0])
        else:
            indexes.append(random.randint(0, len(cell_ids) - 1))
        return indexes

    def whole_mask(self, cell_ids, cell_num):
        """
        Mask all of the tokens in the given token list
        get a random index, mark golden truth, and build fake token list

        inputs: token_ids and numerical_feats of a cell
        return: gold labels, fake token list, [SEP] label
        """
        indexes = self.get_mlm_index_whole(cell_ids, cell_num)
        gold = [-1 for _ in cell_ids]
        fake = [cell_id for cell_id in cell_ids]
        for index in indexes:
            gold[index] = cell_ids[index]
            fake[index] = self.get_replace_token(cell_ids[index])
        sep_label = random.sample(cell_ids, 1)[0]
        return gold, fake, sep_label

    def get_context_truth(self, context):
        truths = []
        if context is None:
            # print("context not available.")
            return truths
        else:
            for text in context[: 2]:
                truths.extend(self.chunk_text(text))
            random.shuffle(truths)
        return truths

    def chunk_text(self, text, max_snippet_len=10):
        words = text.strip().split()
        num_snippets = (len(words) + max_snippet_len - 1) // max_snippet_len  # dump last span
        spans = []
        for i in range(num_snippets):
            spans.append(" ".join(words[i * max_snippet_len: (i + 1) * max_snippet_len]))
        return spans


class CtcTokenizer(TableTokenizer):
    def no_sampling(self, token_matrix):
        sampling_mask = [[1 for _ in token_matrix[0]] for _ in token_matrix]
        return sampling_mask

    def check_valid(self, tokens, numbers, in_header, ctc_label):
        if EMP_ID in tokens:
            return 0
        if 2 <= ctc_label <= 4:
            return 1
        if self.wordpiece_tokenizer.default_num not in set(numbers[1:]):  # pure numeric
            if (in_header == True) or (random.random() < self.value_threshold):
                return 1
        elif set([self.wordpiece_tokenizer.default_num]) == set(numbers[1:]):  # pure text
            if (in_header == True) or (random.random() < self.text_threshold):
                return 2
        else:  # halfway
            if (in_header == True) or (random.random() < self.value_threshold):
                return 2
        return 0

    def sampling(self, token_matrix, number_matrix, header_info, label_matrix):
        """mark each cell: '0' as dumped, '1' as input. """
        header_rows, header_columns = header_info
        sampling_mask = [[1 for _ in token_matrix[0]] for _ in token_matrix]

        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                in_header = (irow < header_rows) or (icol < header_columns)
                ctc_label = int(label_matrix[irow][icol])
                cell_valid = self.check_valid(tokens, number_matrix[irow][icol], in_header, ctc_label)
                sampling_mask[irow][icol] = cell_valid
        return sampling_mask

    def init_table_seq(self, root_context=""):
        """Initialize table sequence with CLS_ID at head, add context if provided. """
        context_tokens, context_number = self.tokenize_text(cell_string=root_context, add_separate=False,
                                                            max_cell_len=64)
        token_list = [[CLS_ID] + context_tokens]
        num_list = [[self.wordpiece_tokenizer.default_num] + context_number]
        pos_list = [(self.row_size, self.column_size, [-1] * self.tree_depth, [-1] * self.tree_depth)]
        fmt_list = [self.default_format]
        ind_list = [[-1] + [-2 for _ in context_tokens]]
        label_list = [[-1] + [-1 for ct in context_tokens]]
        cell_num = 1
        seq_len = len(token_list[0])
        return token_list, num_list, pos_list, fmt_list, ind_list, label_list, cell_num, seq_len

    def create_table_seq(self,
                         sampling_matrix, token_matrix, number_matrix, position_lists, format_matrix, label_matrix,
                         sep_or_tok, add_sep=True
                         ):
        top_pos_list, left_pos_list = position_lists
        row_number, column_number = len(token_matrix), len(token_matrix[0])
        if format_matrix is None:
            format_matrix = [[self.default_format for _ in range(column_number)] for _ in range(row_number)]
        token_list, num_list, pos_list, fmt_list, ind_list, label_list, cell_num, seq_len = self.init_table_seq(
            root_context="")

        for irow, token_row in enumerate(token_matrix):
            for icol, token_cell in enumerate(token_row):
                if sampling_matrix[irow][icol] == 0:
                    continue
                token_list.append(token_cell)
                num_list.append(number_matrix[irow][icol])

                cell_len = len(token_cell)
                icell = irow * column_number + icol
                pos_list.append((irow, icol, top_pos_list[icell], left_pos_list[icell]))
                format_vector = []
                for ivec, vec in enumerate(format_matrix[irow][icol]):
                    format_vector.append(min(vec, self.format_range[ivec]) / self.format_range[ivec])
                fmt_list.append(format_vector)
                ind_list.append([cell_num * 2 for _ in range(cell_len)])
                ind_list[-1][0] -= 1

                ctc_label = int(label_matrix[irow][icol])
                if (ctc_label < 2) or (ctc_label > 4):
                    ctc_label = -1
                else:
                    ctc_label -= 2
                label_list.append([-1 for _ in token_cell])
                label_list[-1][sep_or_tok] = ctc_label
                label_list[-1][1 - sep_or_tok] = ctc_label

                seq_len += cell_len
                cell_num += 1
        return token_list, num_list, pos_list, fmt_list, ind_list, label_list


class FPTokenizer(TableTokenizer):
    """ Formula prediction tokenizer."""
    def check_valid(self, tokens, numbers, in_header):
        """
        Check if a cell is valid enough as table input
            remove empty cell, in both header and data regions
            for all-number cell, keep in header (do not mask), sampling from data
            for half-text&number cell, keep in header (can mask), sampling from data
            for text cell, keep in header (can mask), sampling from data (resp.)
        """
        if EMP_ID in tokens:
            return 0
        if self.wordpiece_tokenizer.default_num not in set(numbers[1:]):  # pure numeric
            if (in_header == True) or (random.random() < self.value_threshold):
                return 1
        elif set([self.wordpiece_tokenizer.default_num]) == set(numbers[1:]):  # pure text
            if (in_header == True) or (random.random() < self.text_threshold):
                return 2
        else:  # halfway
            if (in_header == True) or (random.random() < self.value_threshold):
                return 2
        return 0

    def init_table_seq(self, root_context):
        """Initialize table sequence with CLS_ID at head, add context if provided. """
        context_tokens, context_number = self.tokenize_text(cell_string=root_context, add_separate=False,
                                                            max_cell_len=8)

        token_list = [[CLS_ID] + context_tokens]
        num_list = [[self.wordpiece_tokenizer.default_num] + context_number]
        pos_list = [(self.row_size, self.column_size, [-1] * self.tree_depth, [-1] * self.tree_depth)]
        fmt_list = [self.default_format]
        ind_list = [[-1] + [-2 for _ in context_tokens]]
        formula_label = [[0 for _ in token_list[0]]]
        candi_cell_token_mask = [[0 for _ in token_list[0]]]
        cell_num = 1
        seq_len = len(token_list[0])
        return token_list, num_list, pos_list, fmt_list, ind_list, formula_label, candi_cell_token_mask, cell_num, seq_len

    def get_text_choices(self, context_truths, length_left, max_pair_num=3):
        choice_label_pairs = []
        context_length, num = 0, 0
        for truth_text in context_truths:
            truth_token, truth_num = self.tokenize_text(truth_text, True, 20)
            disturb_text = self.context_repo.pop()  # pair each truth with a disturb
            disturb_token, disturb_num = self.tokenize_text(disturb_text)
            context_length += len(truth_token) + len(disturb_token)
            if (context_length > length_left) or (num > max_pair_num):
                self.context_repo.append(disturb_text)  # push the disturb back if not used
                break
            num += 1
            choice_label_pairs.append(((truth_token, truth_num), 1))
            choice_label_pairs.append(((disturb_token, disturb_num), 0))

        new_repo = random.sample(context_truths, num)
        self.context_repo.extend(new_repo)
        random.shuffle(choice_label_pairs)
        return choice_label_pairs

    def check_empty(self, tokens):
        """ Check if tokens of a cell is empty."""
        return EMP_ID in tokens

    def find_cells_in_matrix(self, tokens, types, table_range):
        """ Convert cell excel coord to matrix coord. e.g. 'B2' -> (0, 0)"""
        range_coords = []
        table_top_left_cell = table_range.split(':')[0]
        for tok, tok_type in zip(tokens, types):
            if tok_type == 'CELL':
                coord = cell_to_coord(tok, table_top_left_cell)
                range_coords.append(coord)
            else:
                range_coords.append((-1, -1))
        return range_coords

    def fp_preprocess(self, token_matrix, number_matrix, position_lists,
                      header_info, formula_row, formula_col, table_range, formula_info,
                      format_matrix=None, context=None, add_sep=True):
        """
        Preprocessing for formula prediction(fp) task.

        Args:
            token_matrix (:obj:`List[List[List]]`):
                Tokenized table cell token ids in matrix format.
            number_matrix (:obj:`List[List[List]]`):
                Tokenized table cell number features in matrix format.
            position_lists (:obj:`Tuple[List[List], List[List]]`):
                Cell top&left tree positions in table in linearized list format.
            header_info: (:obj:`Tuple[int, int]`):
                Number of rows and columns.
            formula_row (:obj:`int`):
                Row index of the target formula in table.
            formula_col (:obj:`int`):
                Column index of the target formula in table.
            table_range (:obj:`str`):
                The table range in excel format. e.g., "A5:B10"
                It is used for quick conversion of excel cell coordinate and python matrix cell index.
            formula_info (:obj:`Dict[]`):
                Contain all available information about the target formula.
                See details in the code.

        Returns:
            token_list (:obj:`List`): token ids of shape (seq_len,).
            num_list (:obj:`List`): number features of shape (seq_len,).
            indicator (:obj:`List`): indices indicating the cells that tokens belong to of shape (seq_len,).
            format_list (:obj:`List`): format features of shape (seq_len,).
            formula_label (:obj:`List`): formula labels of shape (seq_len,).
                1 for formula cell tokens(including [sep]), 0 for the others.
            complete_sketch (:obj:`List`): formula sketch ids of shape (len(formula tokens),).
            candi_cell_token_mask (:obj:`List`): candidate cell token masks of shape (seq_len,).
                1 for candidate cell tokens that may be referenced, 0 for the others.
            range_label (:obj:`List`): referenced cell token indices of shape (len(formula tokens)-1,).
                Store the token index to be referenced in sequence.
                The length is len(formula tokens)-1 because <START> is not included.
            range_map (:obj:`Dict`): map token index to excel coordinate.
        """
        header_rows, header_columns = header_info
        formula_tokens, formula_token_types = formula_info['FormulaTokens'], formula_info[
            'FormulaTokenTypes']  # w/o. <START> or <END>
        formula_range_coords = self.find_cells_in_matrix(formula_tokens, formula_token_types, table_range)
        # sampling matrix
        sampling_mask = [[0 for _ in token_matrix[0]] for _ in token_matrix]
        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                in_header = (irow < header_rows) or (icol < header_columns)
                same_row_or_col = (formula_row == irow) | (formula_col == icol)
                # if not self.check_empty(tokens) and (in_header or same_row_or_col):  # valid input cell
                if (not self.check_empty(
                        tokens) and in_header) or same_row_or_col:  # empty cells on the same line should also be input since they may be referenced
                    sampling_mask[irow][icol] = 1
                if irow == formula_row and icol == formula_col:  # formula cell
                    sampling_mask[irow][icol] = 2

        # cell coord matrix, map coord to range idx  in 'range_label'
        range_matrix = [[-1 for _ in token_matrix[0]] for _ in token_matrix]
        cnt1, cnt2 = 0, 0
        for idx, (row_id, col_id) in enumerate(formula_range_coords):
            if row_id != -1 and col_id != -1:
                range_matrix[row_id][col_id] = idx
                cnt1 += 1

        # build model input sequence
        top_pos_list, left_pos_list = position_lists
        row_number, column_number = len(token_matrix), len(token_matrix[0])
        # rewrite a fake format_matrix (all default values) for wiki & wdc tables
        if format_matrix is None:
            format_matrix = [[[0. for _ in range(self.num_format_feature)] for _ in range(column_number)] for _ in
                             range(row_number)]

        # get all context chunks
        context_truths = self.get_context_truth(context)
        root_context = ""
        if len(context_truths) > 0:
            root_context = context_truths[0]
            context_truths = context_truths[1:]

        token_list, num_list, pos_list, format_list, \
        indicator, formula_label, candi_cell_token_mask, cell_num, seq_len = self.init_table_seq(
            root_context=root_context)
        range_label = [DEFAULT_RANGE_LABEL] * (len(formula_tokens) + 1)  # w/ <END>, w/o. <START>
        range_map = {}
        formula_flag = False
        for irow, sampling_row in enumerate(sampling_mask):
            for icol, mark in enumerate(sampling_row):
                if mark == 0:  # dumped cell
                    continue
                icell = irow * column_number + icol
                cell_num += 1
                tokens = token_matrix[irow][icol]
                number = number_matrix[irow][icol]
                cell_len = len(tokens)
                if mark == 2:  # formula data cell
                    token_list.append([SEP_ID, FORMULA_ID])
                    num_list.append([self.wordpiece_tokenizer.default_num] * 2)
                    formula_label.append([1, 1])
                    formula_flag = True
                    cell_len = 2
                else:  # valid cell
                    token_list.append(tokens)
                    num_list.append(number)
                    formula_label.append([0 for _ in tokens])

                if range_matrix[irow][icol] != -1:
                    range_label[range_matrix[irow][icol]] = seq_len
                    str_excel_coord = formula_tokens[range_matrix[irow][icol]]
                    range_map[seq_len] = str_excel_coord  # for visualizing prediction in fp_head
                    cnt2 += 1
                candi_cell_token_mask.append([1] + [0] * (cell_len - 1))  # only label [sep]

                # add corresponding format vector
                format_vector = []
                for ivec, vec in enumerate(format_matrix[irow][icol]):
                    format_vector.append(min(vec, self.format_range[ivec]) / self.format_range[ivec])
                format_list.append(format_vector)
                pos_list.append((irow, icol, top_pos_list[icell], left_pos_list[icell]))
                indicator.append([cell_num * 2 for _ in range(cell_len)])
                if add_sep:
                    assert (cell_len > 1) or (token_list[-1] == [CLS_ID]), "Mini cell: {}".format(token_list[-1])
                    indicator[-1][0] -= 1
                seq_len += cell_len

        if not formula_flag:
            raise InputMissingError(f"Some input missing in formula {formula_info['A1']}.")

        complete_sketch = [self.fp_tok2id('<START>')]  # with both <START> and <END>
        for i in range(len(formula_tokens)):
            tok, tok_type = formula_tokens[i], formula_token_types[i]
            if tok_type == 'CELL':
                complete_sketch.append(self.fp_tok2id('<RANGE>'))
            elif tok_type == 'STRING':
                complete_sketch.append(self.fp_tok2id('C-STR'))
            elif tok_type == 'NUMBER':
                complete_sketch.append(self.fp_tok2id('C-NUM'))
            elif tok_type == 'BOOL':
                complete_sketch.append(self.fp_tok2id('C-BOOL'))
            elif tok_type == 'SPECIAL':
                complete_sketch.append(self.fp_tok2id(':'))
            else:
                try:
                    complete_sketch.append(self.fp_tok2id(tok))
                except Exception as e:
                    # print(f"'{tok}' is not covered in FP Vocab.")
                    complete_sketch.append(self.fp_tok2id('<UNKOP>'))
        complete_sketch.append(self.fp_tok2id('<END>'))

        return token_list, num_list, pos_list, format_list, indicator, \
               formula_label, complete_sketch, candi_cell_token_mask, range_label, range_map

    def fp_toks2ids(self, toks):
        return [self.fp_tok2id(t) for t in toks]

    def fp_ids2toks(self, ids):
        return [self.fp_id2tok(i) for i in ids]

    def fp_tok2id(self, tok):
        return FP_VOCAB[tok]

    def fp_id2tok(self, id):
        return REV_FP_VOCAB[id]

    def pprint_position_list(self, top_pos_list, left_pos_list, sampling_mask):
        m, n = len(sampling_mask), len(sampling_mask[0])
        print('top pos')
        for i in range(len(top_pos_list)):
            top_coord = top_pos_list[i]
            print(f"{top_coord}", end=';;')
            if i % n == n - 1:
                print()
        print('left pos')
        for i in range(len(left_pos_list)):
            left_coord = left_pos_list[i]
            print(f"{left_coord}", end=';;')
            if i % n == n - 1:
                print()

    def pprint_sampling_matrix(self, mat):
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                cell = mat[i][j]
                print(cell, end=';;')
            print()

    def get_context_truth(self, context):
        truths = []
        if context is None:
            # print("context not available.")
            return truths
        else:
            for text in context[: 2]:
                truths.extend(self.chunk_text(text))
            random.shuffle(truths)
        return truths

    def chunk_text(self, text, max_snippet_len=10):
        words = text.strip().split()
        num_snippets = (len(words) + max_snippet_len - 1) // max_snippet_len  # dump last span
        spans = []
        for i in range(num_snippets):
            spans.append(" ".join(words[i * max_snippet_len: (i + 1) * max_snippet_len]))
        return spans


class FortapTokenizer(TableTokenizer):
    """Formula pretrain tokenizer."""
    def check_valid(self, tokens, numbers, in_header):
        """
        Check if a cell is valid enough as table input
            remove empty cell, in both header and data regions
            for all-number cell, keep in header (do not mask), sampling from data
            for half-text&number cell, keep in header (can mask), sampling from data
            for text cell, keep in header (can mask), sampling from data (resp.)
        """
        if EMP_ID in tokens:
            return 0
        if self.wordpiece_tokenizer.default_num not in set(numbers[1:]):  # pure numeric
            if (in_header == True) or (random.random() < self.value_threshold):
                return 1
        elif set([self.wordpiece_tokenizer.default_num]) == set(numbers[1:]):  # pure text
            if (in_header == True) or (random.random() < self.text_threshold):
                return 2
        else:  # halfway
            if (in_header == True) or (random.random() < self.value_threshold):
                return 2
        return 0

    def init_table_seq(self, context_tokens):
        """Initialize table sequence with CLS_ID at head, add context if provided. """
        token_list = [[CLS_ID] + context_tokens]
        num_list = [[self.wordpiece_tokenizer.default_num for _ in token_list[0]]]
        pos_list = [
            [(self.row_size, self.column_size, [-1] * self.tree_depth, [-1] * self.tree_depth) for _ in token_list[0]]]
        fmt_list = [[self.default_format for _ in token_list[0]]]
        ind_list = [[-1] + [-2 for _ in context_tokens]]
        mlm_label = [[self.default_mlm_label for _ in token_list[0]]]
        sr_list = [[DEFAULT_SR_LABEL for _ in token_list[0]]]
        nr_list = [[DEFAULT_NR_LABEL for _ in token_list[0]]]
        sr_context_list = [[DEFAULT_SR_CONTEXT_LABEL for _ in token_list[0]]]
        op_mlm_label = [[DEFAULT_OP_MLM_LABEL for _ in token_list[0]]]
        range_mlm_label = [[DEFAULT_RANGE_MLM_LABEL for _ in token_list[0]]]
        candi_cell_token_mask = [[0 for _ in token_list[0]]]
        cell_num = 1
        seq_len = len(token_list[0])
        return token_list, num_list, pos_list, fmt_list, ind_list, mlm_label, \
               sr_list, nr_list, sr_context_list, \
               op_mlm_label, range_mlm_label, candi_cell_token_mask, cell_num, seq_len

    def check_empty(self, tokens):
        """ Check if tokens of a cell is empty."""
        return EMP_ID in tokens

    def tag_sr_parent_headers(self, sampling_mask, token_matrix, row, col, formula_row, formula_col,
                              header_rows, header_columns, positive_header_string_set, orient=None,
                              sr_tag=SR_POS_TAG):
        """ Tag sr tag for all parent headers of current cell. """
        success_flag = 0
        anchor_header_row, anchor_header_col = None, None
        formula_header_row, formula_header_col = None, None
        if orient is None:
            if row == formula_row:
                orient = 'top'
            elif col == formula_col:
                orient = 'left'
            else:
                return success_flag, anchor_header_row, anchor_header_col, formula_header_row, formula_header_col
        target_row, target_col = None, None
        if orient == 'top':
            for i in range(row, -1, -1):
                if (sampling_mask[i][col][0] == 1 or sampling_mask[i][col][0] == 2) \
                        and sampling_mask[i][col][1] == NOTUSE_TAG \
                        and i < header_rows:  # the non-empty header next to the data cell
                    sampling_mask[i][col][1] = sr_tag
                    if sr_tag == SR_POS_TAG:
                        sampling_mask[i][col][3] = SR_CONTEXT_REF_HEADER_TAG
                    # positive_header_string_set.add(''.join(token_matrix[i][col]))
                    target_row, target_col = i, col
                    break
        else:
            for j in range(col, -1, -1):
                if (sampling_mask[row][j][0] == 1 or sampling_mask[row][j][0] == 2) \
                        and sampling_mask[row][j][1] == NOTUSE_TAG \
                        and j < header_columns:
                    sampling_mask[row][j][1] = sr_tag
                    if sr_tag == SR_POS_TAG:
                        sampling_mask[row][j][3] = SR_CONTEXT_REF_HEADER_TAG
                    target_row, target_col = row, j
                    break

        if sr_tag == SR_FORMULA_TAG:
            for i in range(row, -1, -1):
                if (sampling_mask[i][col][0] == 1 or sampling_mask[i][col][0] == 2) \
                        and i < header_rows:  # the non-empty header next to the data cell
                    formula_header_row, formula_header_col = i, col
                    break
            for j in range(col, -1, -1):
                if (sampling_mask[row][j][0] == 1 or sampling_mask[row][j][0] == 2) \
                        and j < header_columns:
                    anchor_header_row, anchor_header_col = row, j
                    break
            if orient == 'left':
                formula_header_row, anchor_header_row = anchor_header_row, formula_header_row
                formula_header_col, anchor_header_col = anchor_header_col, formula_header_col
        if target_row is not None:
            success_flag = 1
        return success_flag, anchor_header_row, anchor_header_col, formula_header_row, formula_header_col

    def tuta_pretrain_sampling(self, token_matrix, number_matrix, header_info,
                               max_disturb_num=10, disturb_prob=0.2, clc_rate=0.3):
        header_rows, header_columns = header_info
        sampling_mask = [[1 for _ in token_matrix[0]] for _ in token_matrix]
        divide_prob = random.random()
        divide_prob = 0.4 + divide_prob / 5

        mask = []
        valid_cell_count = 0
        blank_buckets = [set() for _ in range(header_rows + header_columns + 2)]
        # coarse marking
        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                in_header = (irow < header_rows) or (icol < header_columns)
                cell_valid = self.check_valid(tokens, number_matrix[irow][icol], in_header)
                valid_cell_count += cell_valid
                if cell_valid == 0:
                    sampling_mask[irow][icol] = 0
                elif cell_valid > 1:
                    prob = random.random()
                    if prob < (disturb_prob * divide_prob):  # mask
                        mask.append((irow, icol))
                    else:  # blank or none
                        if irow < header_rows:
                            if icol < header_columns:
                                blank_buckets[0].add((irow, icol))
                            else:
                                blank_buckets[1 + irow].add((irow, icol))
                        elif icol < header_columns:
                            blank_buckets[header_rows + 1 + icol].add((irow, icol))
                        else:
                            blank_buckets[-1].add((irow, icol))

        max_disturb_num = min(max_disturb_num, int(disturb_prob * valid_cell_count))
        # refine mask marks
        mask_num = min(len(mask), int(max_disturb_num * divide_prob))
        mask = random.sample(mask, mask_num)
        for irow, icol in mask:
            sampling_mask[irow][icol] = 3

        # refine blank marks
        blank = []
        for bucket in blank_buckets:
            if len(bucket) == 0:
                continue
            if (len(bucket) < 6) and (random.random() < clc_rate):
                blank.extend(random.sample(bucket, 1))
            else:
                bucket_blank_num = min(max(1, int(len(bucket) * clc_rate)), 3)
                bucket_blank = random.sample(bucket, bucket_blank_num)
                blank.extend(bucket_blank)
        blank_num = min(len(blank), max_disturb_num - mask_num)
        if blank_num > 1:
            blank = random.sample(blank, blank_num)
            for irow, icol in blank:
                sampling_mask[irow][icol] = 2
        return sampling_mask

    def find_cells_in_matrix(self, tokens, types, table_range):
        """ Convert cell excel coord to matrix coord. e.g. 'B2' -> (0, 0)"""
        range_coords = []
        table_top_left_cell = table_range.split(':')[0]
        for i, (tok, tok_type) in enumerate(zip(tokens, types)):
            if tok_type == 'CELL':
                if len(tok) > 6:  # an ugly check for xlparser misparse string into cell
                    types[i] = 'STRING'
                    range_coords.append((-1, -1))
                    continue
                try:
                    coord = cell_to_coord(tok, table_top_left_cell)
                    range_coords.append(coord)
                except Exception as e:
                    # print(f"Error in cell to coord '{tok}': {e}")
                    types[i] = 'STRING'
                    range_coords.append((-1, -1))
            else:
                range_coords.append((-1, -1))
        return range_coords

    def generate_context(self, formula_header_tokens, anchor_header_tokens):
        """Generate fake context based on formula header and anchor header."""
        max_context_length = 63
        if formula_header_tokens[0] == SEP_ID:
            formula_header_tokens = formula_header_tokens[1:]
        if anchor_header_tokens[0] == SEP_ID:
            anchor_header_tokens = anchor_header_tokens[1:]
        num_random_tokens = random.randint(1, 10)
        random_tokens = random.sample(list(range(1997, 29613)), num_random_tokens)
        dice1 = random.random()
        if dice1 < 0.5:
            formula_header_tokens, anchor_header_tokens = anchor_header_tokens, formula_header_tokens
        dice2 = random.random()
        if dice2 < 0.7:
            context_tokens = formula_header_tokens + anchor_header_tokens
        else:
            context_tokens = formula_header_tokens + random_tokens + anchor_header_tokens
        return context_tokens[:max_context_length]

    def pretrain_preprocess(self, token_matrix, number_matrix, position_lists,
                            header_info, formula_row, formula_col, table_range, formula_info,
                            format_matrix=None, context=None, file_path=None, add_sep=True,
                            max_disturb_num=10, disturb_prob=0.2, sr_neg_pos_prop=3, formula_tag_prob=0.8,
                            op_mlm_prob=0.5, max_formula_length=30, formula_mlm_prop=1.0):
        """ Preprocessing for pretrain tasks.
        Returns:
            sr_label: (b, seq_len)
                -1 -> not used
                0 -> negative pairs with formula header
                1 -> positive pairs with formula header
            nr_label: (b, seq_len)
                'DEFAULT_NR_LABEL' -> not used
                label of ground truth operator
            sr_context_label: (b, seq_len)
                -1 -> not used
                0 -> normal valid cell
                1 -> anchor header(s), path of headers, for entity link
                2 -> ref header(s), for ref ability
                3 -> ref data cell(s), for ref ability
                4 -> formula data cell, for speculating
            op_mlm_label: (b, seq_len)
                -1 -> not used
                label of ground truth operator
            range_mlm_label: (b, seq_len)
                0 -> not used
                label of ground truth reference cell index in input sequence
            candi_cell_token_mask: (b, seq_len)
                0 -> not candidate cell for formula mlm
                1 -> candidate cell for formula mlm
        """

        header_rows, header_columns = header_info
        sr_dict, nr_dict = formula_info['sr_dict'], formula_info['nr_dict']
        formula_tokens, formula_token_types = formula_info['FormulaTokens'], formula_info[
            'FormulaTokenTypes']  # w/o. <START> or <END>
        if len(formula_tokens) > max_formula_length:
            # print("Formula exceed max_formula_length.")
            return None
        formula_range_coords = self.find_cells_in_matrix(formula_tokens, formula_token_types, table_range)
        tuta_sampling_matrix = self.tuta_pretrain_sampling(token_matrix, number_matrix, header_info,
                                                           max_disturb_num, disturb_prob)
        sampling_mask = [[[0, NOTUSE_TAG, NOTUSE_TAG, SR_CONTEXT_NORMAL_TAG] \
                          for _ in token_matrix[0]]
                         for _ in token_matrix]
        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                in_header = (irow < header_rows) or (icol < header_columns)
                same_row_or_col = (formula_row == irow) | (formula_col == icol)
                if not self.check_empty(tokens) and (in_header or same_row_or_col):
                    sampling_mask[irow][icol][0] = 1
                if irow == formula_row and icol == formula_col:
                    sampling_mask[irow][icol][0] = 2

        # cell coord matrix, map coord to range idx  in 'range_label'
        range_matrix = [[-1 for _ in token_matrix[0]] for _ in token_matrix]
        for idx, (row_id, col_id) in enumerate(formula_range_coords):
            try:
                range_matrix[row_id][col_id] = idx
            except:
                pass

        # set sr positive tag, set nr tag, set sr_context tag
        orient = estimate_sr_orient(sr_dict, formula_row, formula_col)
        positive_header_string_set = set()
        num_sr_pos_cells, num_nr_cells = 0, 0
        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                if irow == formula_row and icol == formula_col:  # formula header cell
                    ret, anchor_header_row, anchor_header_col, formula_header_row, formula_header_col \
                        = self.tag_sr_parent_headers(sampling_mask, token_matrix, irow, icol,
                                                     formula_row, formula_col, header_rows, header_columns,
                                                     positive_header_string_set, orient=orient, sr_tag=SR_FORMULA_TAG)
                    if formula_header_row is None:
                        return None
                    sampling_mask[formula_header_row][formula_header_col][3] = SR_CONTEXT_FORMULA_HEADER_TAG
                    if anchor_header_row is not None:
                        sampling_mask[anchor_header_row][anchor_header_col][3] = SR_CONTEXT_ANCHOR_HEADER_TAG
                    sampling_mask[irow][icol][3] = SR_CONTEXT_FORMULA_CELL_TAG
                elif (irow, icol) in sr_dict:  # sr positive ref header cell
                    ret, _, _, _, _ = self.tag_sr_parent_headers(sampling_mask, token_matrix, irow, icol, formula_row,
                                                                 formula_col, header_rows, header_columns,
                                                                 positive_header_string_set, sr_tag=SR_POS_TAG)
                    sampling_mask[irow][icol][3] = SR_CONTEXT_REF_DATA_CELL_TAG
                    num_sr_pos_cells += ret
                if (irow, icol) in nr_dict:
                    sampling_mask[irow][icol][2] = nr_dict[(irow, icol)]
                    num_nr_cells += 1

        # set sr negative tag
        num_sr_neg_cells, stop_flag = 0, False
        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                if (irow, icol) not in sr_dict:
                    if (irow == formula_row and orient == 'top') or (icol == formula_col and orient == 'left'):
                        ret, _, _, _, _ = self.tag_sr_parent_headers(sampling_mask, token_matrix, irow, icol,
                                                                     formula_row, formula_col, header_rows,
                                                                     header_columns,
                                                                     positive_header_string_set, orient=orient,
                                                                     sr_tag=SR_NEG_TAG)
                        num_sr_neg_cells += ret
                        if num_sr_neg_cells >= num_sr_pos_cells * sr_neg_pos_prop:
                            stop_flag = True
                            break
            if stop_flag:
                break

        # build fake context
        formula_header_tokens = token_matrix[formula_header_row][formula_header_col]
        if anchor_header_row is not None:
            anchor_header_tokens = token_matrix[anchor_header_row][anchor_header_col]
        else:
            anchor_header_tokens = [SEP_ID]
        if formula_header_tokens == [SEP_ID, VAL_ID] \
                or EMP_ID in formula_header_tokens:  # not useful as formula header
            return None
        context_or_inner_dice = random.random()
        if 0 <= context_or_inner_dice < 0.5:
            context_or_inner = 'CONTEXT'
        else:
            context_or_inner = 'INNER'
        fdice = random.random()
        if 0 <= fdice < 0.4:  # tasks: sr|nr|formula_mlm|mlm + 50% sr_context
            formula_input_type = 'FORMULA'
        elif 0.4 <= fdice < 0.7:  # tasks: sr|nr|mlm + 50% sr_context
            formula_input_type = 'TAG'
        else:  # tasks: sr|nr|sr_context|mlm
            formula_input_type = 'TOKEN'
        context_tokens = self.generate_context(formula_header_tokens, anchor_header_tokens)
        if formula_input_type != 'TOKEN' and context_or_inner == 'INNER':
            context_tokens = []

        # build model input sequence
        top_pos_list, left_pos_list = position_lists
        row_number, column_number = len(token_matrix), len(token_matrix[0])
        # rewrite a fake format_matrix (all default values) for wiki & wdc tables
        if format_matrix is None:
            format_matrix = [[[0. for _ in range(self.num_format_feature)] for _ in range(column_number)] for _ in
                             range(row_number)]

        token_list, num_list, pos_list, format_list, indicator, \
        mlm_label, sr_label, nr_label, sr_context_label, \
        op_mlm_label, range_mlm_label, candi_cell_token_mask, \
        cell_num, seq_len = \
            self.init_table_seq(context_tokens=context_tokens)
        formula_flag, sr_ref_flag, sr_formula_flag, nr_op_flag = [False] * 4
        rev_range_map = {}
        fcell_index = None
        for irow, sampling_row in enumerate(sampling_mask):
            for icol, _ in enumerate(sampling_row):
                mark_tuta, mark_formula = tuta_sampling_matrix[irow][icol], sampling_mask[irow][icol]
                if mark_tuta == 0 and mark_formula[0] == 0:
                    continue
                icell = irow * column_number + icol
                # check if has null position
                if (max(top_pos_list[icell]) == -1) or (max(left_pos_list[icell]) == -1):
                    if mark_formula[0] == 2 \
                            or (formula_input_type == 'FORMULA' and range_matrix[irow][
                        icol] != -1):  # necessary cell, skip the whole sample
                        return None
                    else:  # skip unnecessary cell
                        continue
                cell_num += 1
                tokens = token_matrix[irow][icol]
                number = number_matrix[irow][icol]
                cell_len = len(tokens)
                cell_mlm_label = [self.default_mlm_label] * cell_len
                cell_sr_label = [DEFAULT_SR_LABEL] * cell_len
                cell_nr_label = [DEFAULT_NR_LABEL] * cell_len
                cell_sr_context_label = [DEFAULT_SR_CONTEXT_LABEL] * cell_len
                cell_op_mlm_label = [DEFAULT_OP_MLM_LABEL] * cell_len
                cell_range_mlm_label = [DEFAULT_RANGE_MLM_LABEL] * cell_len
                cell_format, format_vector = [], []
                for ivec, vec in enumerate(format_matrix[irow][icol]):
                    format_vector.append(min(vec, self.format_range[ivec]) / self.format_range[ivec])
                cell_format = [format_vector] * cell_len
                cell_pos = [(irow, icol, top_pos_list[icell], left_pos_list[icell])] * cell_len
                cell_indicator = [cell_num * 2] * cell_len
                if mark_formula[0] == 2:  # formula data cell
                    if formula_input_type == 'TAG':
                        tokens = [SEP_ID, FORMULA_ID]
                        number = [self.wordpiece_tokenizer.default_num] * 2
                        cell_mlm_label = [self.default_mlm_label] * 2
                        cell_sr_label = [DEFAULT_SR_LABEL] * 2
                        cell_nr_label = [DEFAULT_NR_LABEL] * 2
                        if context_or_inner == 'CONTEXT':
                            cell_sr_context_label = [SR_CONTEXT_FORMULA_CELL_TAG] * 2
                        else:
                            cell_sr_context_label = [DEFAULT_SR_CONTEXT_LABEL] * 2
                        cell_op_mlm_label = [DEFAULT_OP_MLM_LABEL] * 2
                        cell_range_mlm_label = [DEFAULT_RANGE_MLM_LABEL] * 2
                        cell_format, format_vector = [], []
                        for ivec, vec in enumerate(format_matrix[irow][icol]):
                            format_vector.append(min(vec, self.format_range[ivec]) / self.format_range[ivec])
                        cell_format = [format_vector] * 2
                        cell_pos = [(irow, icol, top_pos_list[icell], left_pos_list[icell])] * 2
                        cell_indicator = [cell_num * 2] * 2
                        cell_len = 2
                    elif formula_input_type == 'FORMULA':
                        tokens, number, cell_format, cell_pos, cell_mlm_label, cell_sr_label, cell_nr_label, \
                        cell_sr_context_label, cell_op_mlm_label, cell_range_mlm_label, cell_indicator \
                            = [], [], [], [], [], [], [], [], [], [], []  # placeholder
                        cell_len = len(formula_tokens) + 1
                        fcell_index = cell_num - 1
                    elif formula_input_type == 'TOKEN':
                        cell_sr_context_label = [SR_CONTEXT_FORMULA_CELL_TAG] * 2 + \
                                                [DEFAULT_SR_CONTEXT_LABEL] * (cell_len - 2)
                    formula_flag = True
                elif mark_formula[0] == 1:  # formula pretrain cell
                    if mark_formula[1] != NOTUSE_TAG:  # sr task cell
                        cell_sr_label = [mark_formula[1]] + [DEFAULT_SR_LABEL for _ in range(len(tokens) - 1)]
                        if mark_formula[1] == SR_FORMULA_TAG:
                            sr_formula_flag = True
                        else:
                            sr_ref_flag = True
                    if mark_formula[2] != NOTUSE_TAG:  # nr task cell
                        cell_nr_label = [mark_formula[2]] + [DEFAULT_NR_LABEL for _ in range(len(tokens) - 1)]
                        nr_op_flag = True
                    if mark_formula[3] > DEFAULT_SR_CONTEXT_LABEL \
                            and (
                            formula_input_type == 'TOKEN' or context_or_inner == 'CONTEXT'):  # sr context task cell
                        cell_sr_context_label = [mark_formula[3]] * 2 + \
                                                [DEFAULT_SR_CONTEXT_LABEL] * (
                                                            cell_len - 2)  # use both [sep] and cell tokens

                elif mark_tuta == 1:  # not formula pretrain, valid (in v5, can be used for sr_context task)
                    if mark_formula[3] > DEFAULT_SR_CONTEXT_LABEL \
                            and (
                            formula_input_type == 'TOKEN' or context_or_inner == 'CONTEXT'):  # sr context task cell
                        cell_sr_context_label = [mark_formula[3]] * 2 + \
                                                [DEFAULT_SR_CONTEXT_LABEL] * (
                                                            cell_len - 2)  # use both [sep] and cell tokens
                elif mark_tuta == 3:  # mask
                    if add_sep == True:
                        if random.random() < self.wcm_rate:
                            gold, fake, sep_label = self.whole_mask(tokens[1:], number[1:])
                        else:
                            gold, fake, sep_label = self.easy_mask(tokens[1:], number[1:])
                        tokens = [SEP_ID] + fake
                        cell_mlm_label = [sep_label] + gold
                if range_matrix[irow][icol] != -1:
                    str_excel_coord = formula_tokens[range_matrix[irow][icol]]
                    rev_range_map[str_excel_coord] = (cell_num - 1, seq_len)
                token_list.append(tokens)
                num_list.append(number)
                format_list.append(cell_format)
                pos_list.append(cell_pos)
                mlm_label.append(cell_mlm_label)
                sr_label.append(cell_sr_label)
                nr_label.append(cell_nr_label)
                sr_context_label.append(cell_sr_context_label)
                op_mlm_label.append(cell_op_mlm_label)
                range_mlm_label.append(cell_range_mlm_label)
                indicator.append(cell_indicator)
                candi_cell_token_mask.append([1] + [0] * (cell_len - 1))
                if add_sep and indicator[-1]:
                    assert (cell_len > 1) or (token_list[-1] == [CLS_ID]), "Mini cell: {}".format(token_list[-1])
                    indicator[-1][0] -= 1
                seq_len += cell_len

        if not (formula_flag & sr_formula_flag):
            return None

        if formula_input_type == 'FORMULA':
            fcell_tokens, fcell_number, fcell_format, fcell_pos, fcell_indicator, \
            fcell_mlm_label, fcell_sr_label, fcell_nr_label, fcell_sr_context_label, \
            fcell_op_mlm_label, fcell_range_mlm_label \
                = self.encode_formula(formula_tokens, formula_token_types, formula_range_coords,
                                      formula_row, formula_col, fcell_index, column_number, rev_range_map,
                                      number_matrix, num_list,
                                      format_matrix, format_list,
                                      pos_list, top_pos_list, left_pos_list,
                                      context_or_inner, op_mlm_prob, formula_mlm_prop)
            token_list[fcell_index][:] = fcell_tokens
            num_list[fcell_index][:] = fcell_number
            pos_list[fcell_index][:] = fcell_pos
            format_list[fcell_index][:] = fcell_format
            indicator[fcell_index][:] = fcell_indicator
            mlm_label[fcell_index][:] = fcell_mlm_label
            sr_label[fcell_index][:] = fcell_sr_label
            nr_label[fcell_index][:] = fcell_nr_label
            sr_context_label[fcell_index][:] = fcell_sr_context_label
            op_mlm_label[fcell_index][:] = fcell_op_mlm_label
            range_mlm_label[fcell_index][:] = fcell_range_mlm_label

        return token_list, num_list, pos_list, format_list, indicator, mlm_label, \
               sr_label, nr_label, sr_context_label, \
               op_mlm_label, range_mlm_label, candi_cell_token_mask

    def encode_formula(self, formula_tokens, formula_token_types, formula_range_coords,
                       formula_row, formula_col, fcell_index, column_number, rev_range_map,
                       number_matrix, num_list,
                       format_matrix, format_list,
                       pos_list, top_pos_list, left_pos_list,
                       context_or_inner, op_mlm_prob, formula_mlm_prop):
        """ Encode formula and add pretrain labels."""
        frow, fcol = formula_row, formula_col
        fcell = frow * column_number + fcol
        format_vector = []
        for ivec, vec in enumerate(format_matrix[frow][fcol]):
            format_vector.append(min(vec, self.format_range[ivec]) / self.format_range[ivec])
        fcell_tokens, fcell_number, fcell_format, fcell_pos, fcell_mlm_label, \
        fcell_sr_label, fcell_nr_label, fcell_sr_context_cell, \
        fcell_op_mlm_label, fcell_range_mlm_label \
            = [SEP_ID], [self.wordpiece_tokenizer.default_num], [format_vector], \
              [(frow, fcol, top_pos_list[fcell], left_pos_list[fcell])], [self.default_mlm_label], \
              [DEFAULT_SR_LABEL], [DEFAULT_NR_LABEL], [DEFAULT_SR_CONTEXT_LABEL], \
              [DEFAULT_OP_MLM_LABEL], [DEFAULT_RANGE_MLM_LABEL]
        if context_or_inner == 'CONTEXT':
            fcell_sr_context_cell = [SR_CONTEXT_FORMULA_CELL_TAG]
        op_mlm_flag = (random.random() < op_mlm_prob)
        for i, (tok, tok_type, coord) in enumerate(zip(formula_tokens, formula_token_types, formula_range_coords)):
            if tok_type == 'CELL':
                token_id = FP_ENCODE_VOCAB[f'[<RANGE>]']
                ref_cell_index, ref_seq_index = rev_range_map.get(tok, (-1, -1))
                nonexist_ref_flag = (ref_cell_index == -1 and ref_seq_index == -1)
                self_ref_flag = not num_list[ref_cell_index]
                invalid_ref_flag = nonexist_ref_flag | self_ref_flag
                if invalid_ref_flag:
                    fcell_tokens.append(token_id)
                    fcell_number.append(number_matrix[frow][fcol][
                                            1])  # List[tuple], index [1] because [0] is for [sep] number, which is default
                    fcell_format.append(format_vector)  # List[List]
                    fcell_pos.append((frow, fcol, top_pos_list[fcell], left_pos_list[fcell]))  # List[tuple]
                    fcell_op_mlm_label.append(DEFAULT_OP_MLM_LABEL)
                    fcell_range_mlm_label.append(DEFAULT_RANGE_MLM_LABEL)
                else:
                    mask_dice = random.random()
                    if i == len(formula_tokens) - 1:
                        remain_cells = 0
                    else:
                        remain_cells = sum([1 if t == 'CELL' else 0 for t in formula_token_types[i + 1:]])
                    mask_flag = not op_mlm_flag \
                                and (mask_dice < formula_mlm_prop or remain_cells == 0)
                    if not mask_flag:  # not mask, can use ref emb
                        fcell_tokens.append(token_id)
                        fcell_number.append(num_list[ref_cell_index][1])
                        fcell_format.append(format_list[ref_cell_index][0])
                        fcell_pos.append(pos_list[ref_cell_index][0])
                        # fcell_number.append(number_matrix[frow][fcol][
                        #                         1])  # List[tuple], index [1] because [0] is for [sep] number, which is default
                        # fcell_format.append(format_vector)  # List[List]
                        # fcell_pos.append((frow, fcol, top_pos_list[fcell], left_pos_list[fcell]))  # List[tuple]
                        fcell_op_mlm_label.append(DEFAULT_OP_MLM_LABEL)
                        fcell_range_mlm_label.append(DEFAULT_RANGE_MLM_LABEL)
                    else:  # mask
                        fcell_tokens.append(token_id)
                        fcell_number.append(number_matrix[frow][fcol][
                                                1])  # List[tuple], index [1] because [0] is for [sep] number, which is default
                        fcell_format.append(format_vector)  # List[List]
                        fcell_pos.append((frow, fcol, top_pos_list[fcell], left_pos_list[fcell]))  # List[tuple]
                        fcell_op_mlm_label.append(DEFAULT_OP_MLM_LABEL)
                        fcell_range_mlm_label.append(DEFAULT_RANGE_MLM_LABEL)
                        mask_method_dice = random.random()
                        if 0 < mask_method_dice <= 0.8:
                            mask_token_id = MASK_ID
                        elif 0.8 < mask_method_dice <= 0.9:
                            mask_token_id = token_id
                        else:
                            mask_token_id = random.randint(0, len(self.vocab) - 1)
                        fcell_tokens[i+1] = mask_token_id
                        fcell_range_mlm_label[i+1] = ref_seq_index
            else:
                if tok_type == 'STRING':
                    token_id = FP_ENCODE_VOCAB['[C-STR]']
                elif tok_type == 'NUMBER':
                    token_id = FP_ENCODE_VOCAB['[C-NUM]']
                elif tok_type == 'BOOL':
                    token_id = FP_ENCODE_VOCAB['[C-BOOL]']
                elif f'[{tok}]' in FP_ENCODE_VOCAB:
                    token_id = FP_ENCODE_VOCAB[f'[{tok}]']
                else:
                    token_id = FP_ENCODE_VOCAB['[<UNKOP>]']
                fcell_tokens.append(token_id)
                fcell_number.append(number_matrix[frow][fcol][1])
                fcell_format.append(format_vector)
                fcell_pos.append((frow, fcol, top_pos_list[fcell], left_pos_list[fcell]))
                fcell_op_mlm_label.append(DEFAULT_OP_MLM_LABEL)
                fcell_range_mlm_label.append(DEFAULT_RANGE_MLM_LABEL)
                if i == len(formula_tokens) - 1:
                    remain_ops = 0
                else:
                    remain_ops = sum([1 if t in ['FUNC', 'OP'] else 0 for t in formula_token_types[i + 1:]])
                mask_dice = random.random()
                mask_flag = tok_type in ['FUNC', 'OP'] \
                            and op_mlm_flag \
                            and (mask_dice < formula_mlm_prop or remain_ops == 0)
                if mask_flag:
                    mask_method_dice = random.random()
                    if 0 < mask_method_dice <= 0.8:
                        mask_token_id = MASK_ID
                    elif 0.8 < mask_method_dice <= 0.9:
                        mask_token_id = token_id
                    else:
                        mask_token_id = random.randint(0, len(self.vocab) - 1)
                    fcell_tokens[i+1] = mask_token_id
                    fcell_op_mlm_label[i+1] = token_id - FP_START_ID  # 0-indexed [0, 40]
            fcell_mlm_label.append(self.default_mlm_label)
            fcell_sr_label.append(DEFAULT_SR_LABEL)
            fcell_nr_label.append(DEFAULT_NR_LABEL)
            if context_or_inner == 'CONTEXT' and len(fcell_sr_context_cell) < 2:
                fcell_sr_context_cell.append(SR_CONTEXT_FORMULA_CELL_TAG)
            else:
                fcell_sr_context_cell.append(DEFAULT_SR_CONTEXT_LABEL)
        fcell_indicator = [(fcell_index + 1) * 2 for _ in range(len(formula_tokens) + 1)]
        fcell_indicator[0] -= 1

        return fcell_tokens, fcell_number, fcell_format, fcell_pos, fcell_indicator, \
               fcell_mlm_label, fcell_sr_label, fcell_nr_label, fcell_sr_context_cell, \
               fcell_op_mlm_label, fcell_range_mlm_label

    def pprint_position_list(self, top_pos_list, left_pos_list, sampling_mask):
        m, n = len(sampling_mask), len(sampling_mask[0])
        print('top pos')
        for i in range(len(top_pos_list)):
            top_coord = top_pos_list[i]
            print(f"{top_coord}", end=';;')
            if i % n == n - 1:
                print()
        print('left pos')
        for i in range(len(left_pos_list)):
            left_coord = left_pos_list[i]
            print(f"{left_coord}", end=';;')
            if i % n == n - 1:
                print()

    def pprint_sampling_matrix(self, mat):
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                cell = mat[i][j]
                print(cell, end=';;')
            print()

    def get_replace_token(self, token_id):
        prob = random.random()
        if prob < 0.8:
            return MASK_ID
        elif prob < 0.9:
            return random.randint(1996, 29611)
        else:
            return token_id

    def get_mlm_index(self, cell_ids, cell_num):
        """ Select one token in list, prefer text over punctuations over numbers. """
        nonum_indexes, text_indexes = [], []
        for ii, (ids, num) in enumerate(zip(cell_ids, cell_num)):
            if self.wordpiece_tokenizer.default_num == num:
                nonum_indexes.append(ii)
                if (ids not in self.punc_ids):
                    text_indexes.append(ii)
        if len(text_indexes) > 0:
            index = random.sample(text_indexes, 1)[0]
        elif len(nonum_indexes) > 0:
            index = random.sample(nonum_indexes, 1)[0]
        else:
            index = random.randint(0, len(cell_ids) - 1)
        return index

    def easy_mask(self, cell_ids, cell_num):
        """
        Mask only one token in the given token list.
        get a random index, mark golden truth, and build fake token list

        inputs: token_ids and numerical_feats of a cell
        return: gold labels, fake token list, [SEP] label
        """
        index = self.get_mlm_index(cell_ids, cell_num)
        gold = [-1 for _ in cell_ids]
        gold[index] = cell_ids[index]
        fake = cell_ids[: index] + [self.get_replace_token(cell_ids[index])] + cell_ids[index + 1:]
        sep_label = -1  # cell_ids[index]
        return gold, fake, sep_label

    def get_mlm_index_whole(self, cell_ids, cell_num):
        """ Record all viable tokens in list, prefer text over punctuations over numbers. """
        nonum_indexes, text_indexes = [], []
        indexes = []
        for ii, (ids, num) in enumerate(zip(cell_ids, cell_num)):
            if self.wordpiece_tokenizer.default_num == num:
                nonum_indexes.append(ii)
                if (ids not in self.punc_ids):
                    text_indexes.append(ii)
        if len(text_indexes) > 0:
            indexes = text_indexes
        elif len(nonum_indexes) > 0:
            indexes.append(random.sample(nonum_indexes, 1)[0])
        else:
            indexes.append(random.randint(0, len(cell_ids) - 1))
        return indexes

    def whole_mask(self, cell_ids, cell_num):
        """
        Mask all of the tokens in the given token list
        get a random index, mark golden truth, and build fake token list

        inputs: token_ids and numerical_feats of a cell
        return: gold labels, fake token list, [SEP] label
        """
        indexes = self.get_mlm_index_whole(cell_ids, cell_num)
        gold = [-1 for _ in cell_ids]
        fake = [cell_id for cell_id in cell_ids]
        for index in indexes:
            gold[index] = cell_ids[index]
            fake[index] = self.get_replace_token(cell_ids[index])
        sep_label = random.sample(cell_ids, 1)[0]
        return gold, fake, sep_label


class InputMissingError(Exception):
    def __init__(self, message):
        pass
