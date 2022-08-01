# Pre-training data samples
For Spreadsheet, Wiki Table, and Wdc Table, respectively.

For spreadsheet tables, the json file contains keys as follows:

T: cell text (explicitly revealed to humans, e.g.,58.50%)

V: cell value (internally stored in spreadsheets, e.g., 58.5)

NS: number string (internally stored in spreadsheets,e.g., 0.00%)

DT: data type (internally stored in spreadsheets, text=0,number=1,data_time=2,percentage=3,currency=4,others=5)

HF: if has formula, 0 or 1 (if a cell contains a formula in a spreadsheet)

A1: formula string with A1 form (absolute cell reference)

R1: formula string with R1C1 form (relative cell reference)

LB: if has left border, 0 or 1

TB: if has top border, 0 or 1

BB: if has bottom border, 0 or 1

RB: if has right border, 0 or 1

BC: if has non-white background color, 0 or 1

FC: if has non-black font color, 0 or 1

FB: if has font bold, 0 or 1

I: if has font italic, 0 or 1

HA: horizontal alignment (center=0, center_across_selection=1,distributed=2,fill=3,general=4,justify=5,left=6,right=7)

VA: vertical alignment (top=0,center=1,bottom=2,justify=3,distributed=4)

O: operator of the formula, only used in ForTap

FormulaNode: parse tree of the formula, only used in ForTap

FormulaType: indicating if the formula reference one or more cells in other spreadsheets or files. Only used in ForTap
