#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize tables (the reader output) to generate model inputs
with pre-processing for three pre-training objectives:
    MLM: Masked Language Model
    CLC: Cell Level Cloze
    TCR: Table Context Retrieval

"""

from typing import Dict, List
import os
import random
import collections
import unicodedata
from utils import UNZIPS


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

# corresponding token-strings
VAL_TOKEN = "[VAL]"
EMP_TOKEN = "[EMP]"

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"


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
        for index,line in enumerate(reader):
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
            if stype:    # digit span (with , and .)
                stext = self.remove_comma_in_number(stext)
                slist, stype_list = self.split_by_dot(stext)
                text_list.extend(slist)
                type_list.extend(stype_list)
            else:        # text span
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
        stripped_text = whitespace_split(" ".join(stripped_text))   # not necessary
        assert len(stripped_text) == len(stripped_type), "sizes of text and type don't match."
        return stripped_text, stripped_type   # list of texts without whitespaces

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
        if len(parts) > 2:    # not a value
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
            elif char in self.digit_set: # current item is digit
                if digit_flag is True:   # previous item also a digit
                    output[-1].append(char)
                else:                    # alter from text to digit
                    type_list.append(digit_flag)  # type mark of previous span
                    digit_flag = True
                    output.append([char])
            else:                        # current item is text
                if digit_flag is False:  # previous item also text
                    output[-1].append(char)
                else:                    # alter from digit to text
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
        if word_type is True:    # digit span
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
            [index-1 for index in range(1000, 1004)] + 
            [index-1 for index in range(1025, 1037)] + 
            [index-1 for index in range(1064, 1996)] + 
            [index-1 for index in range(29613, 30522)]
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
        self.clc_rate= args.clc_rate
        self.wcm_rate = args.wcm_rate   # whole-cell-mask rate

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
        repository.extend( [line.strip() for line in lines] )
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
        if self.wordpiece_tokenizer.default_num not in set(numbers[1: ]):  # pure numeric
            if (in_header == True) or (random.random() < self.value_threshold):
                return 1
        elif set([self.wordpiece_tokenizer.default_num]) == set(numbers[1: ]):  # pure text
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
                    if prob < (disturb_prob * divide_prob):    # mask
                        mask.append( (irow, icol) )
                    else:                                      # blank or none
                        if irow < header_rows:
                            if icol < header_columns:
                                blank_buckets[0].add( (irow, icol) )
                            else:
                                blank_buckets[1+irow].add( (irow, icol) )
                        elif icol < header_columns:
                            blank_buckets[header_rows+1+icol].add( (irow, icol) )
                        else:
                            blank_buckets[-1].add( (irow, icol) )

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
                blank.extend( random.sample(bucket, 1) )
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
        context_tokens, context_number = self.tokenize_text(cell_string=root_context, add_separate=False, max_cell_len=8)
        if len(context_tokens) > 0:   # do context mlm if available
            gold_labels, masked_tokens, _ = self.easy_mask(context_tokens, context_number)
            token_list = [ [CLS_ID] + masked_tokens ]
            mlm_label = [ [self.default_mlm_label] + gold_labels ]
        else:
            token_list = [ [CLS_ID] ]
            mlm_label = [ [self.default_mlm_label] ]
        num_list = [ [self.wordpiece_tokenizer.default_num] + context_number ]
        pos_list = [ (self.row_size, self.column_size, self.default_tree_position, self.default_tree_position) ] 
        format_list = [ self.default_format ]
        indicator = [ [-1] + [-2 for _ in context_tokens] ]
        clc_label = [ [self.default_clc_label for _ in token_list[0]] ]
        cell_num = 1
        seq_len = len(token_list[0])
        return token_list, num_list, pos_list, format_list, indicator, mlm_label, clc_label, cell_num, seq_len
    
    def get_text_choices(self, context_truths, length_left, max_pair_num=3):
        choice_label_pairs = []
        context_length, num = 0, 0
        for truth_text in context_truths:
            truth_token, truth_num = self.tokenize_text(truth_text, True, 20)
            disturb_text = self.context_repo.pop()   # pair each truth with a disturb
            disturb_token, disturb_num = self.tokenize_text(disturb_text)
            context_length += len(truth_token) + len(disturb_token)
            if (context_length > length_left) or (num > max_pair_num):
                self.context_repo.append(disturb_text)  # push the disturb back if not used
                break
            num += 1
            choice_label_pairs.append( ((truth_token, truth_num), 1) )
            choice_label_pairs.append( ((disturb_token, disturb_num), 0) )
            
        new_repo = random.sample(context_truths, num)
        self.context_repo.extend( new_repo )
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
            format_matrix = [[[0. for _ in range(self.num_format_feature)] for _ in range(column_number)] for _ in range(row_number)]

        # get all context chunks
        context_truths = self.get_context_truth(context)
        root_context = ""
        if len(context_truths) > 0:
            root_context = context_truths[0]
            context_truths = context_truths[1: ]            

        token_list, num_list, pos_list, format_list, \
        indicator, mlm_label, clc_label, cell_num, seq_len = self.init_table_seq(root_context=root_context)
        paste_token_list, paste_num_list, paste_clc_label = [], [], []
        
        for irow, sampling_row in enumerate(sampling_matrix):
            for icol, mark in enumerate(sampling_row):
                if mark == 0:    # dumped
                    continue
                cell_num += 1
                tokens = token_matrix[irow][icol]
                number = number_matrix[irow][icol]
                cell_len = len(tokens)
                if mark == 1:    # normal
                    token_list.append(tokens)
                    num_list.append(number)
                    mlm_label.append( [self.default_mlm_label for _ in tokens] )
                    clc_label.append( [self.default_clc_label for _ in tokens] )
                elif mark == 2: # blank
                    paste_token_list.append(tokens)
                    paste_num_list.append(number)
                    if add_sep == True:
                        token_list.append( [SEP_ID, EMP_ID] )
                        num_list.append( [self.wordpiece_tokenizer.default_num] * 2 )
                        mlm_label.append( [self.default_mlm_label, self.default_mlm_label] )
                        clc_label.append( [-seq_len, -seq_len-1] )
                        assert cell_len > 1
                        real_clc_label = [seq_len, seq_len+1] + [0] * (cell_len-2)
                        paste_clc_label.append( real_clc_label )
                        cell_len = 2
                    else:
                        token_list.append( [EMP_ID] )
                        num_list.append( [self.wordpiece_tokenizer.default_num] )
                        mlm_label.append( [self.default_mlm_label] )
                        clc_label.append( [-seq_len] )
                        assert cell_len > 1
                        real_clc_label = [seq_len] + [0] * (cell_len-1)
                        paste_clc_label.append( real_clc_label )
                        cell_len = 1
                elif mark == 3: # mask
                    if add_sep == True:
                        if random.random() < self.wcm_rate:
                            gold, fake, sep_label = self.whole_mask(tokens[1: ], number[1: ])
                        else:
                            gold, fake, sep_label = self.easy_mask(tokens[1: ], number[1: ])
                        token_list.append( [SEP_ID] + fake )
                        mlm_label.append([sep_label] + gold)
                    else:
                        gold, fake = self.easy_mask(tokens, number)
                        token_list.append( fake )
                        mlm_label.append( gold )
                    num_list.append(number)                    
                    clc_label.append( [self.default_clc_label for _ in tokens] )
                
                # add corresponding format vector
                format_vector = []
                for ivec, vec in enumerate(format_matrix[irow][icol]):
                    format_vector.append( min(vec, self.format_range[ivec]) / self.format_range[ivec] )
                format_list.append( format_vector )

                icell = irow * column_number + icol
                # check if has null position
                if (max(top_pos_list[icell]) == -1) or (max(left_pos_list[icell]) == -1):
                    return None
                pos_list.append( (irow, icol, top_pos_list[icell], left_pos_list[icell]) )
                indicator.append( [cell_num*2 for _ in range(cell_len)] )
                if add_sep:
                    assert (cell_len > 1) or (token_list[-1] == [CLS_ID]), "Mini cell: {}".format(token_list[-1])
                    indicator[-1][0] -= 1
                seq_len += cell_len

        paste_position = (self.row_size, self.column_size, [-1] * self.tree_depth, [-1] * self.tree_depth)
        for tokens in paste_token_list:
            cell_num += 1
            seq_len += len(tokens)
            # ramdom positions for pasted cells to enlarge the attention distance with other cells
            paste_position = (self.row_size, self.column_size, [random.randint(0, 31)] * self.tree_depth, [random.randint(0, 31)] * self.tree_depth)
            
            pos_list.append(paste_position)
            format_list.append(self.default_format)
            mlm_label.append( [self.default_mlm_label for _ in tokens] )
            paste_ind = [cell_num*2] * len(tokens)
            if add_sep:
                paste_ind[0] -= 1
            indicator.append( [-dd for dd in paste_ind] )
        token_list.extend(paste_token_list)
        num_list.extend(paste_num_list)
        clc_label.extend(paste_clc_label)

        # add table-level context choices
        tcr_label = []
        for clc in clc_label:
            tcr_label.append( [-1 for _ in clc] )
        
        context_choice_label_pairs = self.get_text_choices(context_truths, self.max_seq_len - seq_len)
        # adjust number of choices based on current sequence length
        for (token, number), label in context_choice_label_pairs:
            cell_num += 1
            token_list.append(token)
            num_list.append(number)
            mlm_label.append( [self.default_mlm_label for _ in token] )
            paste_position = (self.row_size, self.column_size, [random.randint(0, 31)] * self.tree_depth, [random.randint(0, 31)] * self.tree_depth)
            pos_list.append(paste_position)
            format_list.append(self.default_format)
            indicator.append([-cell_num*2 for _ in token])
            if add_sep:
                indicator[-1][0] += 1
            clc_label.append( [self.default_clc_label for _ in token] )
            tcr_label.append( [-1 for _ in token] )
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
        fake = cell_ids[: index] + [self.get_replace_token(cell_ids[index])] + cell_ids[index + 1: ]
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
                truths.extend( self.chunk_text(text) )
            random.shuffle(truths)            
        return truths
    
    def chunk_text(self, text, max_snippet_len=10):
        words = text.strip().split()
        num_snippets = (len(words) + max_snippet_len - 1) // max_snippet_len  # dump last span
        spans = []
        for i in range(num_snippets):
            spans.append( " ".join(words[i*max_snippet_len: (i+1)*max_snippet_len]) )
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
        if self.wordpiece_tokenizer.default_num not in set(numbers[1: ]):  # pure numeric
            if (in_header == True) or (random.random() < self.value_threshold):
                return 1
        elif set([self.wordpiece_tokenizer.default_num]) == set(numbers[1: ]):  # pure text
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
        context_tokens, context_number = self.tokenize_text(cell_string=root_context, add_separate=False, max_cell_len=64)
        token_list = [ [CLS_ID] + context_tokens ]
        num_list = [ [self.wordpiece_tokenizer.default_num] + context_number ]
        pos_list = [ (self.row_size, self.column_size, [-1] * self.tree_depth, [-1] * self.tree_depth) ] 
        fmt_list = [ self.default_format ]
        ind_list = [ [-1] + [-2 for _ in context_tokens] ]
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
        token_list, num_list, pos_list, fmt_list, ind_list, label_list, cell_num, seq_len = self.init_table_seq(root_context="")

        for irow, token_row in enumerate(token_matrix):
            for icol, token_cell in enumerate(token_row):
                if sampling_matrix[irow][icol] == 0:
                    continue
                token_list.append( token_cell )
                num_list.append( number_matrix[irow][icol] )

                cell_len = len(token_cell)
                icell = irow * column_number + icol
                pos_list.append( (irow, icol, top_pos_list[icell], left_pos_list[icell]) )
                format_vector = []
                for ivec, vec in enumerate(format_matrix[irow][icol]):
                    format_vector.append( min(vec, self.format_range[ivec]) / self.format_range[ivec] )
                fmt_list.append( format_vector )
                ind_list.append( [cell_num*2 for _ in range(cell_len)] )
                ind_list[-1][0] -= 1

                ctc_label = int(label_matrix[irow][icol])
                if (ctc_label < 2) or (ctc_label > 4):
                    ctc_label = -1
                else:
                    ctc_label -= 2
                label_list.append( [-1 for _ in token_cell] )
                label_list[-1][sep_or_tok] = ctc_label
                label_list[-1][1-sep_or_tok] = ctc_label

                seq_len += cell_len
                cell_num += 1
        return token_list, num_list, pos_list, fmt_list, ind_list, label_list


    
class TtcTokenizer(TableTokenizer):
    """Adapted tokenizer for Table Type Classification."""

    def __init__(self, args):
        super(TtcTokenizer, self).__init__(args)
    
    def check_valid(self, tokens, numbers, in_header):
        if EMP_ID in tokens:
            return 0

        if self.wordpiece_tokenizer.default_num not in set(numbers[1: ]):  # pure numeric
            if (in_header == True) or (random.random() < self.value_threshold):
                return 1
        elif set([self.wordpiece_tokenizer.default_num]) == set(numbers[1: ]):  # pure text
            if (in_header == True) or (random.random() < self.text_threshold):
                return 2
        else:  # halfway
            if (in_header == True) or (random.random() < self.value_threshold):
                return 2
        return 0

    def sampling(self, token_matrix, number_matrix):
        """Mark each cell: '0' as dumped, other cases as input. """
        sampling_mask = [[1 for cell in row] for row in token_matrix]
        header_rows, header_columns = 1, 1

        for irow, token_row in enumerate(token_matrix):
            for icol, tokens in enumerate(token_row):
                # in_header = (irow < header_rows) or (icol < header_columns)
                cell_valid = self.check_valid(tokens, number_matrix[irow][icol], in_header=True)
                sampling_mask[irow][icol] = cell_valid

        return sampling_mask
    
    def init_table_lists(self, root_text: str = "") -> Dict:
        """Initialize table sequences with CLS_ID at head for TTC prediction, add context if provided."""
        context_tokens, context_number = self.tokenize_text(cell_string=root_text, add_separate=False, max_cell_len=32)

        token_list = [ [CLS_ID] +  context_tokens ]
        num_list = [ [self.wordpiece_tokenizer.default_num] + context_number ]
        pos_list = [(self.row_size, self.column_size, [-1]*self.tree_depth, [-1]*self.tree_depth)]
        fmt_list = [ self.default_format ]
        ind_list = [ [-1] + [-2 for _ in context_tokens] ]
        
        cell_num = 1
        seq_len = len(token_list[0])

        return token_list, num_list, pos_list, fmt_list, ind_list, cell_num, seq_len
    
    def create_table_lists(
        self, 
        string_matrix: List[List[str]], 
        top_position_list: List[List[List[int]]], 
        left_position_list: List[List[List[int]]],
        title: str, 
        label: int, 
        format_matrix: List = None, 
        add_separate: bool = True, 
        **kwargs
    ): 
        token_list, num_list, pos_list, fmt_list, ind_list, cell_num, seq_len = self.init_table_lists(title)

        nrows, ncols = len(string_matrix), len(string_matrix[0])
        token_matrix, number_matrix = self.tokenize_string_matrix(string_matrix, add_separate)
        sampling_matrix = self.sampling(token_matrix, number_matrix)
        assert len(token_matrix) == len(number_matrix) == len(sampling_matrix) == nrows
        assert len(token_matrix[0]) == len(number_matrix[0]) == len(sampling_matrix[0]) == ncols
        if format_matrix is None:
            format_matrix = [[self.default_format for _ in range(ncols)] for _ in range(nrows)]
        
        for irow, token_row in enumerate(token_matrix):
            for icol, token_cell in enumerate(token_row):
                if sampling_matrix[irow][icol] == 0: continue
    
                token_list.append(token_cell)
                num_list.append(number_matrix[irow][icol])

                icell = irow * ncols + icol
                pos_list.append( (irow, icol, top_position_list[icell], left_position_list[icell]) )

                cell_len = len(token_cell)
                format_vector = []
                for ivec, vec in enumerate(format_matrix[irow][icol]):
                    format_vector.append( min(vec, self.format_range[ivec]) / self.format_range[ivec] )
                fmt_list.append( format_vector )
                ind_list.append( [cell_num*2 for _ in range(cell_len)] )
                ind_list[-1][0] -= 1

                seq_len += cell_len
                cell_num += 1
        
        return token_list, num_list, pos_list, fmt_list, ind_list, label

    @staticmethod
    def table_lists_to_seq(lists, target):
        """Serialize lists of loaded samples to model input sequences."""
        token_list, num_list, pos_list, fmt_list, ind_list, ttc_label = lists
        token_id, num_mag, num_pre, num_top, num_low = [], [], [], [], []
        token_order, pos_row, pos_col, pos_top, pos_left = [], [], [], [], []
        fmt_vec, indicator = [], []

        for tokens, num_feats, (r, c, t, l), fmt, ind in zip(token_list, num_list, pos_list, fmt_list, ind_list):
            cell_len = len(tokens)
            cell_len = min(8, cell_len)

            token_id.extend(tokens[:cell_len])
            num_mag.extend([f[0] for f in num_feats[:cell_len]])
            num_pre.extend([f[1] for f in num_feats[:cell_len]])
            num_top.extend([f[2] for f in num_feats[:cell_len]])
            num_low.extend([f[3] for f in num_feats[:cell_len]])
            token_order.extend([ii for ii in range(cell_len)])
            pos_row.extend([r for _ in range(cell_len)])
            pos_col.extend([c for _ in range(cell_len)])
            entire_top = UNZIPS[target](t)
            pos_top.extend([entire_top for _ in range(cell_len)])
            entire_left = UNZIPS[target](l)
            fmt_vec.extend( [fmt for _ in range(cell_len)] )
            pos_left.extend([entire_left for _ in range(cell_len)])
            indicator.extend(ind[:cell_len])
        
        if len(token_id) > 256: return None
        
        return (
            token_id, num_mag, num_pre, num_top, num_low, 
            token_order, pos_row, pos_col, pos_top, pos_left, 
            fmt_vec, indicator, ttc_label
        )
