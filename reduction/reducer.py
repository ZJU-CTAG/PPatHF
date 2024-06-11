import difflib
import logging
import warnings
import re
from unidiff import PatchSet, PatchedFile, Hunk
from typing import Dict, Set, Union, List, Any
from fcu import FunctionCompareUtilities
from reducerConfig import enough_alpha_threshold


class Reducer:
    def __init__(self,
                 tolerant=False):
        
        # set to True (give warnings instead of throw errors & abort)
        self.tolerant = tolerant
        
    @staticmethod
    def get_modified_lines(mfile: PatchedFile):
        deleted_line_nos = set()
        added_line_nos = set()
        for hunk in mfile:
            for line in hunk:
                if line.line_type == '-':
                    deleted_line_nos.add(line.source_line_no)
                elif line.line_type == '+':
                    added_line_nos.add(line.target_line_no)
        return deleted_line_nos, added_line_nos

    @staticmethod
    def line_no_map(src_no: int, src_unique_set: Set[int], tgt_unique_set: Set[int]) -> Union[int, None]:
        if src_no in src_unique_set:
            return None
        tgt_no = src_no
        tgt_no -= len([_ for _ in src_unique_set if _ < src_no])
        for tgt_unique_line_no in sorted(list(tgt_unique_set)):
            if tgt_unique_line_no > tgt_no:
                break
            tgt_no += 1
        return tgt_no

    def expand_line_nos_set(self, from_set: Set[int], to_set: Set[int], deleted_set: Set[int], added_set: Set[int]) \
            -> Set[int]:
        to_set_new = to_set.copy()
        for line_no in from_set:
            tgt_line_no = self.line_no_map(line_no, deleted_set, added_set)
            if tgt_line_no is None:
                continue
            to_set_new.add(tgt_line_no)
        return to_set_new

    @staticmethod
    def _enough_alpha(string):
        return len([c for c in string if c.isalpha()]) >= enough_alpha_threshold

    def remove_macro_function_definition(self, code: str) -> str:

        def _is_valid_parentheses(s: str):
            stack = []
            for c in s:
                if c == '(':
                    stack.append(c)
                elif c == ')':
                    if not stack or stack[-1] != '(':
                        return False
                    stack.pop()
            return not stack

        lines = code.splitlines(keepends=True)
        
        if len(lines) < 3:
            if self.tolerant:
                warnings.warn(f"input func is too short: {code}")
                return code
            else:
                raise ValueError(f"input func is too short: {code}")
    
        macro_lines = []
        for line_num in range(1, len(lines) - 1):
            if lines[line_num].upper() == lines[line_num] and \
                    lines[line_num - 1].strip().endswith(')') and \
                    lines[line_num + 1].strip().startswith('{') and \
                    self._enough_alpha(lines[line_num]) and \
                    _is_valid_parentheses(lines[line_num]):
                macro_lines.append(line_num)
        for macro_line in macro_lines:
            logging.debug(f"remove_macro_function_definition(): {lines[macro_line]}")
            lines[macro_line] = "\n"
        return "".join(lines)

    def remove_macro_function_parameter_list(self, func: str) -> str:
        match = re.search(r'\((.*?)\)', func, flags=re.DOTALL)
        if match:
            content_inside_brackets = match.group(1)
            lines = content_inside_brackets.split("\n")
            new_lines = []
            for line in lines:
                items = line.split(",")
                new_items = []
                for item in items:
                    words = item.split(" ")
                    for i in range(len(words)):
                        if words[i].upper() == words[i] and self._enough_alpha(words[i]):
                            words[i] = ""
                    words = [word for word in words if word != ""]
                    new_items.append(" ".join(words))
                new_lines.append(",".join(new_items))
            modified_content = "\n".join(new_lines)
            logging.debug(f"remove_macro_function_parameter_list(): {content_inside_brackets} -> {modified_content}")
            return func.replace(match.group(0), f'({modified_content})')

        if self.tolerant:
            warnings.warn(f"Fail to retrieve the parameter list of the input func: {func}")
            return func

    @staticmethod
    def remove_preprocs(code: str) -> str:
        lines = code.splitlines(keepends=True)
        for i in range(len(lines)):
            if lines[i].strip().startswith("#"):
                lines[i] = "\n"
        return "".join(lines)

    def func_standardization(self, func: str):
        return self.remove_preprocs(
            self.remove_macro_function_definition(self.remove_macro_function_parameter_list(func)))

    def do_recovering(self, reduced_func: str, removed_pieces: dict) -> str:
        fcu = FunctionCompareUtilities()

        def retrieve_placeholders(ast_node):
            if ast_node.type == "comment":
                pattern = r'placeholder_(\d+)'
                comment_contents = fcu.retrieve_bytes_as_string(reduced_func_no_macros, ast_node.start_byte,
                                                                ast_node.end_byte)
                match = re.search(pattern, comment_contents)
                if match:
                    matched_id = match.group(1)
                    placeholder_locations[matched_id] = (ast_node.start_point, ast_node.end_point)
            else:
                for child in ast_node.children:
                    retrieve_placeholders(child)

        reduced_func_no_macros = self.func_standardization(reduced_func)
        ast = fcu.parse(reduced_func_no_macros.encode('utf-8'))
        placeholder_locations = dict()
        retrieve_placeholders(ast.root_node)

        loc_str_pairs = []
        for placeholder_index in removed_pieces.keys():
            if placeholder_index not in placeholder_locations:
                if self.tolerant:
                    warnings.warn(f"placeholder_{placeholder_index} not found!")
                    continue
                else:
                    raise ValueError(f"placeholder_{placeholder_index} not found!")
            removed_piece = removed_pieces[placeholder_index]
            removed_location = placeholder_locations[placeholder_index]
            loc_str_pairs.append((removed_location, removed_piece))
        loc_str_pairs = sorted(loc_str_pairs, key=lambda pair: pair[0], reverse=True)

        lines = reduced_func.splitlines(keepends=True)
        for loc, piece in loc_str_pairs:
            if loc[0][0] != loc[1][0]:
                raise ValueError("a placeholder must be in one line")
            lines[loc[0][0]] = lines[loc[0][0]][:loc[0][1]] + piece + lines[loc[0][0]][loc[1][1]:]

        return "".join(lines)
