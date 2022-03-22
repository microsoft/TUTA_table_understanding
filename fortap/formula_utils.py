""" Utils for formula processing."""

import re
from openpyxl.utils import column_index_from_string, get_column_letter

from constants import (
    NR_AGGR_TO_INDEX,
    AGGRS,
    UNARY_OPS,
    BIN_OPS,
    PERCENT,
    RANGE_SEP
)


def tag_formula_token_type(formula_token_list):
    """ Tag formula tokens. Types include:
    1. cell
    2. op/func
    3. string/number/bool
    4. special, i.e. ':'
    """
    formula_tokens, formula_token_types = [], []
    for tok in formula_token_list:
        if tok in UNARY_OPS:
            # new_tok = 'u' + tok
            new_tok = tok  # currently regard u'+' and '+' as the same op
            type = 'OP'
        elif tok == PERCENT or tok in BIN_OPS:
            new_tok = tok
            type = 'OP'
        elif 'Function' in tok:
            new_tok = extract_terminal_value(tok, func_flag=True)
            type = 'FUNC'
        elif 'Cell' in tok:
            new_tok = extract_terminal_value(tok)
            new_tok = new_tok.replace('$', '')  # no $ ground truth target
            type = 'CELL'
        elif 'Text' in tok:
            new_tok = extract_terminal_value(tok)
            type = 'STRING'
        elif 'Number' in tok:
            new_tok = extract_terminal_value(tok)
            type = 'NUMBER'
        elif 'Bool' in tok:
            new_tok = extract_terminal_value(tok)
            type = 'BOOL'
        elif tok == RANGE_SEP:
            new_tok = RANGE_SEP
            type = 'SPECIAL'
        else:
            raise ValueError(f"'{tok}' cannot be extracted.")
        formula_tokens.append(new_tok)
        formula_token_types.append(type)
    return formula_tokens, formula_token_types


def convert_to_prefix(formula_string, root):
    """ Convert a formula string to prefix order, which will be tokenized by fp tokenizer."""
    change_op_to_prefix(root)
    formula_token_list = _convert_to_prefix(root)
    formula_tokens, formula_token_types = tag_formula_token_type(formula_token_list)
    return formula_tokens, formula_token_types


def _convert_to_prefix(root, num_same_level_nodes=0):
    """ Convert a formula string to prefix order, which will be tokenized by fp tokenizer."""
    if root['children'] is None:
        result = [root['name']]
        # if extract_terminal_value(root['name']) in UNARY_OPS:  # disntinguish unary '+' '-'
        #     result = ['u-' + root['name']]
        return result
    result = []
    for child in root['children']:
        result.extend(_convert_to_prefix(child, len(root['children'])))
    return result


def change_op_to_prefix(root):
    """ Change binary op and unary op to prefix order in tree."""
    if root['children'] is None:
        return
    if root['name'] == 'FunctionCall':
        children = root['children']
        if children[1]['name'] == PERCENT:
            children[0], children[1] = children[1], children[0]
        elif children[1]['name'] in BIN_OPS:
            children[0], children[1] = children[1], children[0]
    for child in root['children']:
        change_op_to_prefix(child)


def prepare_sr(formula_info, table_range):
    """ Prepare semantic reference pairs.
    Return:
        sr_dict: {coord: sr_label, ...}, i.e. {(0, 1): 0, (3, 5): 1}
    """
    root = formula_info['FormulaNodes']
    ref_cells = get_ref_cells_from_xl_ast(root)
    table_top_left_cell = table_range.split(':')[0]
    sr_dict = construct_sr_dict(ref_cells, table_top_left_cell)
    return sr_dict


def prepare_nr(formula_info, table_range):
    """ Prepare numerical reasoning pairs.
    Return:
        nr_dict: {coord: nr_label, ...}, i.e. {(0, 1): 7, (3, 5): 1}
    """
    root = formula_info['FormulaNodes']
    cell_groups = get_cell_groups_from_xl_ast(root)
    table_top_left_cell = table_range.split(':')[0]
    nr_dict = construct_nr_dict(cell_groups, table_top_left_cell)
    return nr_dict


def construct_sr_dict(ref_cells, table_top_left_cell):
    """ Construct sr dict from reference cells."""
    sr_dict = {}
    ref_cell_coords = cells_to_coords(ref_cells, table_top_left_cell)
    for coord in ref_cell_coords:
        sr_dict[coord] = 1
    return sr_dict


def construct_nr_dict(cell_groups, table_top_left_cell):
    """ Construct nr dict from cell groups. """
    nr_dict = {}
    for operator, groups in cell_groups.items():
        op_idx = NR_AGGR_TO_INDEX[operator]
        for group_idx, group in enumerate(groups, 1):
            if group_idx > 1:  # for the same operator, only use one group in a formula
                break
            for cell in group:
                coord = cell_to_coord(cell, table_top_left_cell)
                nr_dict[coord] = op_idx
    return nr_dict


def get_cell_groups_from_xl_ast(root):
    """ Wrapper for getting cell groups from XLParser AST."""
    cell_groups = {}
    _get_cell_groups_from_xl_ast(root, cell_groups)
    return cell_groups


def _get_cell_groups_from_xl_ast(root, cell_groups):
    """ Get cell groups from XLParser AST."""
    if root['children'] is None:
        return
    # ic(root['name'])
    if root['name'] == 'FunctionCall':
        results = explore_func_call(root)
        if results is not None:
            operator, operand_cells = results
            if operator not in cell_groups:
                cell_groups[operator] = []
            cell_groups[operator].append(operand_cells)
    for child in root['children']:
        _get_cell_groups_from_xl_ast(child, cell_groups)


def explore_func_call(root):
    """ Explore 'FunctionCall' to get direct operands."""
    children = root['children']
    if children[0]['name'] == 'FunctionName' and children[1]['name'] == 'Arguments':
        return explore_aggr(root)
    elif children[0]['name'] in UNARY_OPS:
        return explore_unary_op(root)
    elif children[1]['name'] == PERCENT:
        return explore_percent(root)
    elif children[1]['name'] in BIN_OPS:
        return explore_bin_op(root)
    else:
        print("'FunctionCall' doesn't belong to AGGRS/UNARY_OPS/PERCENT/BIN_OPS.")


def explore_aggr(root):
    """ Explore 'FunctionCall' of aggregation."""
    children = root['children']
    function_name_node = children[0]
    # extract aggregation name
    operator = extract_terminal_value(function_name_node['children'][0]['name'], func_flag=True)
    if operator not in AGGRS:
        return None
    # extract operand cells
    operand_cells = []
    arguments_node = children[1]
    for arg_node in arguments_node['children']:
        formula_node = arg_node['children'][0]
        ref_node = formula_node['children'][0]
        if ref_node['name'] != 'Reference':
            return None
        node = ref_node['children'][0]
        if node['name'] == 'ReferenceFunctionCall':
            if node['children'][1]['name'] != ':':
                return None
            else:
                start_cell = ref_to_cell_token(node['children'][0])
                end_cell = ref_to_cell_token(node['children'][2])
                if start_cell is None or end_cell is None:
                    return None
                # cells = expand_cells_from_range(start_cell, end_cell)
                cells = [start_cell, end_cell]
                operand_cells.extend(cells)
        else:
            cell = ref_to_cell_token(ref_node)
            if cell is None:
                return None
            operand_cells.append(cell)
    return operator, operand_cells


def explore_unary_op(root):
    """ Explore 'FunctionCall' of unary operation."""
    children = root['children']
    # extract operator
    operator = children[0]['name']
    # extract operand cells
    operand_cells = []
    formula_node = children[1]
    ref_node = formula_node['children'][0]
    if ref_node['name'] != 'Reference':
        return None
    cell = ref_to_cell_token(ref_node)
    if cell is None:
        return None
    operand_cells.append(cell)
    return 'u' + operator, operand_cells


def explore_percent(root):
    """ Explore 'FunctionCall' of percent."""
    children = root['children']
    # extract operator
    operator = children[1]['name']
    # extract operand cells
    operand_cells = []
    formula_node = children[0]
    ref_node = formula_node['children'][0]
    if ref_node['name'] != 'Reference':
        return None
    cell = ref_to_cell_token(ref_node)
    if cell is None:
        return None
    operand_cells.append(cell)
    return operator, operand_cells


def explore_bin_op(root):
    """ Explore 'FunctionCall' of binary operation."""
    children = root['children']
    # extract operator
    operator = children[1]['name']
    # extract operand cells
    operand_cells = []
    for formula_node in [children[0], children[2]]:
        ref_node = formula_node['children'][0]
        if ref_node['name'] != 'Reference':
            return None
        cell = ref_to_cell_token(ref_node)
        if cell is None:
            return None
        operand_cells.append(cell)
    return operator, operand_cells


def ref_to_cell_token(root):
    """ Directly jump to the leaf cell token from reference. """
    assert root['name'] == 'Reference'
    children = root['children']
    if children[0]['name'] != 'Cell':
        return None
    else:
        terminal_string = children[0]['children'][0]['name']
        cell_string = extract_terminal_value(terminal_string)
        return cell_string


def expand_cells_from_range(start_cell, end_cell):
    """ Expand range like A1:B2 to ['A1', 'A2', 'B1', 'B2']. """
    # TODO
    return


def estimate_sr_orient(sr_dict, formula_row, formula_col):
    """ Estimate header orientation of formula cell in sr task."""
    cnt_top, cnt_left = 0, 0
    for (row, col) in sr_dict:
        if row == formula_row:
            cnt_top += 1
        if col == formula_col:
            cnt_left += 1
    if cnt_top >= cnt_left:
        return 'top'
    else:
        return 'left'


def cells_to_coords(ref_cells, top_left_cell):
    """ From cell string position like 'A2' to matrix format like (1, 0). List format."""
    return [cell_to_coord(c, top_left_cell) for c in ref_cells]


def cell_to_coord(ref_cell, top_left_cell):
    """ From cell string position like 'A2' to matrix format like (1, 0)."""
    ref_cell = ref_cell.replace('$', '')
    row_ref_cell, column_ref_cell = extract_cell_row_column(ref_cell)
    row_top_left_cell, column_top_left_cell = extract_cell_row_column(top_left_cell)
    # ic(row_ref_cell, row_top_left_cell)
    # ic(column_ref_cell, column_top_left_cell)
    assert row_ref_cell >= row_top_left_cell and column_ref_cell >= column_top_left_cell
    return row_ref_cell - row_top_left_cell, column_ref_cell - column_top_left_cell


def coord_to_cell(row, column, top_left_cell):
    """ From matrix format to cell string position."""
    row_top_left_cell, column_top_left_cell = extract_cell_row_column(top_left_cell)
    row_ref_cell = row_top_left_cell + row
    column_ref_cell = column_top_left_cell + column
    assert row_ref_cell >= row_top_left_cell and column_ref_cell >= column_top_left_cell
    excel_cell = str(get_column_letter(column_ref_cell)) + str(row_ref_cell)
    return excel_cell


def extract_cell_row_column(cell):
    """ Extract row and column index from cell string position like 'A2'. """
    row_string, column_string = find_row(cell)[0], find_column(cell)[0]
    row, column = int(row_string), int(column_index_from_string(column_string))
    return row, column


def find_column(cell):
    """ Parse column letter from 'A2'. """
    return re.findall('[a-zA-Z]+', cell)


def find_row(cell):
    """ Parse row number from 'A2'. """
    return re.findall('[0-9]+', cell)


def get_ref_cells_from_xl_ast(root):
    """ Wrapper for getting reference cells from XLParser AST."""
    ref_cells = []
    _get_ref_cells_from_xl_ast(root, ref_cells)
    return ref_cells


def _get_ref_cells_from_xl_ast(root, ref_cells):
    """ Get reference cells from XLParser AST."""
    if root['children'] is None:
        if 'CellToken' in root['name']:
            cell_string = extract_terminal_value(root['name'])
            ref_cells.append(cell_string)
        return
    for child in root['children']:
        _get_ref_cells_from_xl_ast(child, ref_cells)


def extract_terminal_value(string, func_flag=False):
    """ Extract pure value from terminal string. i.e. "CellToken['A2']" -> "A2" """
    start_idx, end_idx = string.find('[') + 2, len(string) - 2
    if func_flag:
        end_idx -= 1
    return string[start_idx: end_idx]
