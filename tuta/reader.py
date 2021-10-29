#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table Reader Classes of spreadsheet, wiki, and wdc tables in JSON format

@in: path of a one-table/multi-table file
@out: strings, positions, and formats as needed by the tokenzier

"""


import json


# %% General reader class
class Reader(object):
    def __init__(self, args):
        self.tree_depth = args.tree_depth
        self.node_degree = args.node_degree
        self.row_size = args.row_size
        self.column_size = args.column_size
    
    def init_position_matrix(self, root, row_number, column_number, is_top):
        """ Initialize position martrix, apply row/column indexes if tree not applicable. """
        init_matrix = [[([-1] * self.tree_depth) for _ in range(column_number)] for _ in range(row_number)]
        if (root is not None) and (len(root["Cd"]) > 0):
            return init_matrix
        
        # if tree not applicable, assign as the most shallow layer
        if is_top and (column_number <= self.node_degree[-1]):    # equivalent of column position embedding
            for irow in range(row_number):
                for icol in range(column_number):
                    init_matrix[irow][icol][-1] = icol
        if (not is_top) and (row_number <= self.node_degree[-1]): # equivalent of row position embedding
            for irow in range(row_number):
                for icol in range(column_number):
                    init_matrix[irow][icol][-1] = irow
        return init_matrix
    
    def get_tree_depth(self, node):
        """ Get the depth of tree, root exclusive. """
        depth = 0
        if (node is None) or (len(node["Cd"]) == 0):
            return depth
        for child in node["Cd"]:
            child_depth = self.get_tree_depth(child)
            depth = max(depth, child_depth)
        return depth + 1

    def get_merge_region(self, cell_row, cell_column, merged_regions):
        """ Get the merged region that the current cell belongs to. """
        for merged_region in merged_regions:
            first_row, first_column = merged_region["FirstRow"], merged_region["FirstColumn"]
            last_row, last_column = merged_region["LastRow"], merged_region["LastColumn"]
            if (first_row <= cell_row <= last_row) and (first_column <= cell_column <= last_column):
                return merged_region
        return None

    def dfs_zipping(self, node, node_position, position_matrix, 
        merged_regions, row_number, column_number, is_top, level):
        """ Depth-first traversal of a tree to get zipped position. """
        if (position_matrix is None) or (node is None):
            return position_matrix   # return None if not valid, else terminate if cell not applicable
        children = node["Cd"]
        if children is None:
            return position_matrix

        children_num = len(children)
        if children_num > 0:  # non-leaf
            if (level >= self.tree_depth) or (children_num > self.node_degree[level]):  # or (node_position[0] > 0):
                # print("Exceed tree size: child_num: {}, current depth: {}".format(children_num, level))
                return None   # invalid tree
            for ichild, child_node in enumerate(children):
                child_irow, child_icol = child_node["RI"], child_node["CI"]
                child_position = node_position[: level] + [ichild] + node_position[level + 1: ]
                if position_matrix is None:
                    return position_matrix
                position_matrix[child_irow][child_icol] = child_position
                position_matrix = self.dfs_zipping(
                    node=child_node, 
                    node_position=child_position, 
                    position_matrix=position_matrix, 
                    merged_regions=merged_regions, 
                    row_number=row_number, 
                    column_number=column_number, 
                    is_top=is_top, 
                    level=level + 1
                )
        else:                 # leaf, trace to the data row/column
            node_irow, node_icol = node["RI"], node["CI"]
            node_merge = self.get_merge_region(node_irow, node_icol, merged_regions)
            # create tracing region
            if is_top:
                srow, erow = node_irow + 1, row_number - 1
                if node_merge is None:
                    scol, ecol = node_icol, node_icol
                else:
                    scol, ecol = node_merge["FirstColumn"], node_merge["LastColumn"]
            else:
                scol, ecol = node_icol + 1, column_number - 1
                if node_merge is None:
                    srow, erow = node_irow, node_irow
                else:
                    srow, erow = node_merge["FirstRow"], node_merge["LastRow"]
            # assign position vectors
            for leaf_irow in range(srow, erow + 1):
                for leaf_icol in range(scol, ecol + 1):
                    position_matrix[leaf_irow][leaf_icol] = node_position
        return position_matrix

    def read_header(self, root, merged_regions, row_number, column_number, is_top):
        """ create lists of top&left tree positions from table. """
        position_matrix = self.init_position_matrix(root, row_number, column_number, is_top)
        header_depth = self.get_tree_depth(root)
        if 0 < header_depth <= self.tree_depth:      # [1, 2, 3, 4]
            root_position = [-1] * self.tree_depth
            level = self.tree_depth - header_depth   # depth_index = [3, 2, 1, 0]
            position_matrix = self.dfs_zipping(
                node=root, 
                node_position=root_position, 
                position_matrix=position_matrix, 
                merged_regions=merged_regions, 
                row_number=row_number, 
                column_number=column_number, 
                is_top=is_top, 
                level=level
            )
        if position_matrix is None:
            return None
        position_list = [position for row in position_matrix for position in row]
        return position_list
        
    def tables_from_bigfile(self, input_path, read_bound=None):
        tables = []
        with open(input_path, "r", encoding='utf-8') as fjson:
            for i, line in enumerate(fjson):
                try:
                    table = json.loads(line.strip())
                    tables.append(table)
                except:
                    print("Fail for table at line: ", i);
                if (read_bound is not None) and (i > read_bound):
                    break
        return tables

    def results_from_bigfile(self, input_path, read_bound=None):
        tables = self.tables_from_bigfile(input_path, read_bound)
        results = []
        for table in tables:
            result = self.result_from_table(table)
            if result is not None:
                results.append(result)
        return results



# %% Readers for specific types of tables
class SheetReader(Reader):
    def info_from_matrix(self, cell_matrix, merged_regions, formula_value_augmentation=False):
        """ 
        Get the string and format matrices. 
        Format features order as:
            0: number of merged row, 
            1: number of merged columns,
            2: if cell has a top border,
            3: if cell has a bottom border,
            4: if cell has a left border,
            5: if cell has a right border,
            6: if cell string of the date type,
            7: if cell string contains formula,
            8: if cell string has bold font,
            9: if background color is white,
            10: if font color is black.
        """
        string_matrix, format_matrix = [], []
        for cell_row in cell_matrix:
            string_matrix.append( [cell["V"] for cell in cell_row] )
            format_matrix.append( [] )
            for cell in cell_row:
                format_cell = [1, 1]
                format_cell.extend([cell[key] for key in ["TB", "BB", "LB", "RB"]])
                format_cell.append( int(cell["DT"] > 0) )
                format_cell.append( cell["HF"] )
                if formula_value_augmentation and cell["HF"] != 0 and str(cell["HF"]) != "0": 
                    cell["V"] = "=# " + cell["V"]
                format_cell.append( cell["FB"] )
                format_cell.append( int((cell["BC"][0] == "#") and (cell["BC"].endswith("ffffff"))) )
                format_cell.append( int((cell["FC"][0] == "#") and (cell["FC"].endswith("000000"))) )

                format_matrix[-1].append(format_cell)
        
        # update merged regions
        for mbox in merged_regions:
            frow, fcol, lrow, lcol = mbox["FirstRow"], mbox["FirstColumn"], mbox["LastRow"], mbox["LastColumn"]
            height, width = lrow - frow + 1, lcol - fcol + 1
            for irow in range(frow, lrow + 1):
                for icol in range(fcol, lcol + 1):
                    format_matrix[irow][icol][0] = height
                    format_matrix[irow][icol][1] = width
        return string_matrix, format_matrix

    def result_from_file(self, json_path):
        """ Load sample (one) table from input path, return info needed by the tokenizer. """
        # Step 1: Load File
        try:    
            with open(json_path, encoding='utf-8') as fjson: 
                table = json.load(fjson) 
        except:
            print("Fail to load json file: ", json_path)
            return None
        cell_matrix = table["Cells"]     # get string and format
        merged_regions = table["MergedRegions"]
        row_number, column_number = len(cell_matrix), len(cell_matrix[0])
        if (row_number > self.row_size) or (column_number > self.column_size):
            print("Fail for extreme sizes: {} rows, {} columns ".format(row_number, column_number))
            return None

        # Step 2: Get Position Lists
        try:
            top_position_list = self.read_header(table["TopTreeRoot"], merged_regions, row_number, column_number, True)
            if top_position_list is None:
                print("Fail to generate top-tree position list")
                top_position_list = self.read_header(None, merged_regions, row_number, column_number, True)
            left_position_list = self.read_header(table["LeftTreeRoot"], merged_regions, row_number, column_number, False)
            if left_position_list is None:
                print("Fail to generate the left-tree position list")
                left_position_list = self.read_header(None, merged_regions, row_number, column_number, False)
        except:
            print("Error in read header. ")
            return None

        # Step 3: Convert to Tokenizer Input
        string_matrix, format_matrix = self.info_from_matrix(cell_matrix, merged_regions)
        header_rows, header_columns = table["TopHeaderRowsNumber"], table["LeftHeaderColumnsNumber"]
        return string_matrix, (top_position_list, left_position_list), (header_rows, header_columns), format_matrix

    

class WikiReader(Reader):
    def result_from_table(self, table):
        string_matrix = table["Texts"]
        title = table["Title"]
        merged_regions = table["MergedRegions"]
        row_number, column_number = len(string_matrix), len(string_matrix[0])
        if (row_number > self.row_size) or (column_number > self.column_size):
            print("Fail for extreme sizes: {} rows, {} columns ".format(row_number, column_number))
            return None

        try:
            top_position_list = self.read_header(table["TopTreeRoot"], merged_regions, row_number, column_number, True)
            if top_position_list is None:
                print("Fail to generate top-tree position list")
                top_position_list = self.read_header(None, merged_regions, row_number, column_number, True)
            left_position_list = self.read_header(table["LeftTreeRoot"], merged_regions, row_number, column_number, False)
            if left_position_list is None:
                print("Fail to generate the left-tree position list")
                left_position_list = self.read_header(None, merged_regions, row_number, column_number, False)
        except:
            print("Error in read header. ")
            return None

        header_rows, header_columns = table["TopHeaderRowsNumber"], table["LeftHeaderColumnsNumber"]
        return string_matrix, (top_position_list, left_position_list), (header_rows, header_columns), title



class WdcReader(Reader):
    def result_from_table(self, table):
        string_matrix = table["Texts"]
        title = table["Title"]
        merged_regions = table["MergedRegions"]
        row_number, column_number = len(string_matrix), len(string_matrix[0])
        if (row_number > self.row_size) or (column_number > self.column_size):
            print("Fail for extreme sizes: {} rows, {} columns ".format(row_number, column_number))
            return None

        try:
            top_position_list = self.read_header(table["TopTreeRoot"], merged_regions, row_number, column_number, True)
            if top_position_list is None:
                print("Fail to generate top-tree position list")
                top_position_list = self.read_header(None, merged_regions, row_number, column_number, True)
            left_position_list = self.read_header(table["LeftTreeRoot"], merged_regions, row_number, column_number, False)
            if left_position_list is None:
                print("Fail to generate the left-tree position list")
                left_position_list = self.read_header(None, merged_regions, row_number, column_number, False)
        except:
            print("Error in read header. ")
            return None

        header_rows, header_columns = table["TopHeaderRowsNumber"], table["LeftHeaderColumnsNumber"]
        return string_matrix, (top_position_list, left_position_list), (header_rows, header_columns), title



# %% Reader dictionary
READERS = {"sheet": SheetReader, "wiki": WikiReader, "wdc": WdcReader}
