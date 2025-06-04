import difflib
import logging
import warnings
import re
from unidiff import PatchSet, PatchedFile, Hunk
from typing import Dict, Set, Union, List, Any
from fcu import FunctionCompareUtilities
from reducerConfig import basic_unit_type, enough_alpha_threshold
from reducerConfig import levenshtein_similarity_threshold_weak as similarity_threshold_weak
from reducerConfig import levenshtein_similarity_threshold_parent as parent_threshold
from reducerConfig import levenshtein_similarity_threshold_ancestors as ancestors_threshold
from reducerConfig import length_ratio_threshold as len_threshold
from jellyfish import levenshtein_distance


class Reducer:
    def __init__(self, 
                 threshold_self=similarity_threshold_weak, 
                 threshold_parent=parent_threshold,
                 threshold_ancestors=ancestors_threshold,
                 tolerant=False):
        self.threshold_self = threshold_self
        self.threshold_parent = threshold_parent
        self.threshold_ancestors = threshold_ancestors
        
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

    def cal_critical_lines(self, before_commit: str, after_commit: str) -> Dict:
        diff = ''.join(difflib.unified_diff(before_commit.splitlines(keepends=True),
                                            after_commit.splitlines(keepends=True),
                                            fromfile="dummy_fname.c", tofile="dummy_fname.c", n=3))

        fps = PatchSet(diff).modified_files
        assert len(fps) != 0, "no file_patch found!"

        mfile = fps[0]
        deleted_line_nos, added_line_nos = self.get_modified_lines(mfile)

        return {
            "diff": diff,
            "del_set": deleted_line_nos,
            "add_set": added_line_nos
        }

    @staticmethod
    def get_removable_trees(ast, critical_line_nos: Set[int]) -> List[Dict]:
        removable_trees = []

        def dfs(root, depth, remove_root=None, path_type="t"):
            nonlocal removable_trees
            start_line = root.start_point[0]
            end_line_inclusive = root.end_point[0]
            my_line_nos_set = set(range(start_line, end_line_inclusive + 1))

            if root.type in basic_unit_type:
                path_type = path_type + root.type[:1]
            
            new_remove_node = None
            if len(my_line_nos_set.intersection(critical_line_nos)) == 0 and root.type in basic_unit_type:
                new_remove_node = {
                    "children": [],
                    "location": ((root.start_point[0], root.start_point[1]), (root.end_point[0], root.end_point[1])),
                    "ast_node": root,
                    "type": root.type,
                    "path_type": path_type,
                    "bytes": root.end_byte - root.start_byte
                }
                if remove_root is None:
                    removable_trees.append(new_remove_node)
                else:
                    remove_root["children"].append(new_remove_node)

            for child in root.children:
                if new_remove_node is not None:
                    dfs(child, depth + 1, new_remove_node, path_type)
                else:
                    dfs(child, depth + 1, remove_root, path_type)

        dfs(ast.root_node, 0)
        return removable_trees

    def get_removable_tree_tuples(self, func_before: str, func_after: str, unique_set_before: Set[int],
                                  unique_set_after: Set[int]):
        def map_code_slice(original_loc, original_unique, target_unique):
            target_line_start = self.line_no_map(original_loc[0][0], original_unique, target_unique)
            target_row_start = original_loc[0][1]
            target_line_end = self.line_no_map(original_loc[1][0], original_unique, target_unique)
            target_row_end = original_loc[1][1]
            assert target_line_start is not None, "look for a critical line while wanting to retrieve removable lines!"
            assert target_line_end is not None, "look for a critical line while wanting to retrieve removable lines!"

            return (target_line_start, target_row_start), (target_line_end, target_row_end)

        def check_location_validness(loc):

            def dfs_rm_tree_check_exist(rm_tree: Dict, check_loc):
                if rm_tree["location"] == check_loc:
                    return rm_tree
                if rm_tree["children"] is None:
                    return None
                for child in rm_tree["children"]:
                    ret = dfs_rm_tree_check_exist(child, check_loc)
                    if ret is not None:
                        return ret
                return None

            target_location = map_code_slice(loc, unique_set_before, unique_set_after)
            for rm_tree_after in rm_trees_after:
                found_loc = dfs_rm_tree_check_exist(rm_tree_after, target_location)
                if found_loc is not None:
                    return found_loc
            return None

        def dfs_rm_tree_find_pairs(rm_tree, before_parent=None, after_parent=None):
            loc = rm_tree["location"]
            match_loc = check_location_validness(loc)
            if match_loc is None:
                for child in rm_tree["children"]:
                    dfs_rm_tree_find_pairs(child, before_parent, after_parent)
            else:
                rm_tree_clone = rm_tree.copy()
                rm_tree_clone["children"] = []
                match_loc_clone = match_loc.copy()
                match_loc_clone["children"] = []
                if before_parent is None:
                    removable_pairs.append((rm_tree_clone, match_loc_clone))
                else:
                    before_parent["children"].append(rm_tree_clone)
                    after_parent["children"].append(match_loc_clone)
                for child in rm_tree["children"]:
                    dfs_rm_tree_find_pairs(child, rm_tree_clone, match_loc_clone)

        fcu = FunctionCompareUtilities()
        removable_pairs = []

        ast_before = fcu.parse(self.func_standardization(func_before).encode('utf-8'))
        rm_trees_before = self.get_removable_trees(ast_before, unique_set_before)

        ast_after = fcu.parse(self.func_standardization(func_after).encode('utf-8'))
        rm_trees_after = self.get_removable_trees(ast_after, unique_set_after)

        for tree in rm_trees_before:
            dfs_rm_tree_find_pairs(tree)

        return removable_pairs

    def mapping_find_match(self, target: dict, ast_mapping_dict: dict, removable_mapping_locations: set,
                           func_before: str, func_mapping: str):
        fcu = FunctionCompareUtilities()

        def normalize(original_string: str) -> str:
            return (re.sub(r'\s+|[{}]', '', original_string)).lower()

        def calculate_similarity(str1, str2):
            if max(len(str1), len(str2)) < 800 or 1 / len_threshold <= len(str1) / len(str2) <= len_threshold:
                distance = levenshtein_distance(str1, str2)
                max_length = max(len(str1), len(str2))
                s = (max_length - distance) / max_length
                return s
            return 0.0

        def check_loc_overlap(loc):
            for mapped_loc in removable_mapping_locations:
                before, after = min(loc, mapped_loc, key=lambda l: l[0]), max(loc, mapped_loc,
                                                                              key=lambda l: l[0])
                if before[1] > after[0]:
                    return True
            return False

        def parent_similarity(ast_node_before, ast_node_mapping, check_ancestors=False):
            def get_parent(current_node):
                parent = current_node.parent

                while True:
                    if not check_ancestors and parent.type in basic_unit_type:
                        return parent
                    assert parent.parent is not None, "no function_definition in get_parent()"

                    if parent.parent.type == "function_definition":
                        assert parent.type == "compound_statement", "son of function_definition not being" \
                                                                    "compound_statement in get_parent()"
                        return parent
                    parent = parent.parent

            def weighted_harmonic_mean(x1, x2, w1, w2):
                weighted_mean = (w1 + w2) / ((w1 / x1) + (w2 / x2))
                return weighted_mean

            parent_before = get_parent(ast_node_before)
            parent_mapping = get_parent(ast_node_mapping)

            if parent_before.type == "compound_statement" and not check_ancestors:
                return 2.0

            parent_str_before_backward = fcu.retrieve_string_slice(func_before, parent_before.start_point,
                                                                   ast_node_before.end_point)
            parent_str_before_forward = fcu.retrieve_string_slice(func_before, ast_node_before.start_point,
                                                                  parent_before.end_point)
            parent_str_mapping_backward = fcu.retrieve_string_slice(func_mapping, parent_mapping.start_point,
                                                                    ast_node_mapping.end_point)
            parent_str_mapping_forward = fcu.retrieve_string_slice(func_mapping, ast_node_mapping.start_point,
                                                                   parent_mapping.end_point)
            parent_pattern_before_backward = normalize(parent_str_before_backward)
            parent_pattern_before_forward = normalize(parent_str_before_forward)
            parent_pattern_mapping_backward = normalize(parent_str_mapping_backward)
            parent_pattern_mapping_forward = normalize(parent_str_mapping_forward)

            similarity_backward = calculate_similarity(parent_pattern_before_backward,
                                                       parent_pattern_mapping_backward)
            similarity_forward = calculate_similarity(parent_pattern_before_forward,
                                                      parent_pattern_mapping_forward)

            if similarity_forward == 0.0 or similarity_backward == 0.0:
                return 0.0

            weight_backward = max(len(parent_pattern_before_backward), len(parent_pattern_mapping_backward))
            weight_forward = max(len(parent_pattern_before_forward), len(parent_pattern_mapping_forward))

            result = weighted_harmonic_mean(similarity_backward, similarity_forward, weight_backward, weight_forward)

            return result

        if target["path_type"] not in ast_mapping_dict:
            return None

        found = []
        target_location = target["location"]
        target_pattern = normalize(fcu.retrieve_string_slice(func_before, target_location[0], target_location[1]))

        for mapping_node in ast_mapping_dict[target["path_type"]]:
            node_pattern = normalize(fcu.retrieve_string_slice(func_mapping, mapping_node["location"][0],
                                                               mapping_node["location"][1]))
            location = mapping_node["location"]
            if check_loc_overlap(location):
                continue

            similarity = calculate_similarity(target_pattern, node_pattern)
            if similarity >= self.threshold_self:
                found.append((similarity, mapping_node))

        if len(found) == 0:
            return None

        evaluate_parent = [(parent_similarity(target["ast_node"], n[1]["ast_node"], False), n[1]) for n in found]
        evaluate_parent = [e for e in evaluate_parent if e[0] > self.threshold_parent]

        if len(evaluate_parent) == 1 and evaluate_parent[0][0] != 2.0:
            evaluate_ancestors = evaluate_parent
        else:
            evaluate_ancestors = [(parent_similarity(target["ast_node"], n[1]["ast_node"], True), n[1])
                                  for n in evaluate_parent]
            evaluate_ancestors = sorted([e for e in evaluate_ancestors if e[0] > self.threshold_ancestors],
                                        key=lambda e: e[0], reverse=True)

        if len(evaluate_ancestors) == 0:
            return None

        best_location = (evaluate_ancestors[0][1]["location"])
        removable_mapping_locations.add(best_location)

        return evaluate_ancestors[0][1]

    def get_removable_tree_triplets(self, func_before: str, func_mapping: str, rm_tree_tuples: List):
        def dfs_for_triplets(rm_tuple):
            ret = self.mapping_find_match(rm_tuple[0], ast_mapping_dict, removable_mapping_locations,
                                          func_before, func_mapping)
            if ret is not None:
                removable_triplets.append((rm_tuple[0], rm_tuple[1], ret))
                return

        removable_triplets = []
        removable_mapping_locations = set()

        fcu = FunctionCompareUtilities()
        ast_mapping = fcu.parse(self.func_standardization(func_mapping).encode('utf-8'))
        ast_mapping_dict = dict()

        removable_trees_mapping = self.get_removable_trees(ast_mapping, set())
        for rm_tree in removable_trees_mapping:
            stack = [rm_tree]
            while stack:
                node = stack.pop()
                if node["path_type"] in ast_mapping_dict:
                    ast_mapping_dict[node["path_type"]].append(node)
                else:
                    ast_mapping_dict[node["path_type"]] = [node]
                stack += node["children"]

        for rm_tree_tuple in rm_tree_tuples:
            dfs_for_triplets(rm_tree_tuple)
        return removable_triplets

    @staticmethod
    def check_merge(func, loc_front, loc_back):
        loc_front, loc_back = min(loc_front, loc_back, key=lambda loc: loc[0]), \
            max(loc_front, loc_back, key=lambda loc: loc[0])
        between = ""
        lines = func.splitlines(keepends=True)
        between += lines[loc_front[1][0]][loc_front[1][1]:]
        for line_no in range(loc_front[1][0] + 1, loc_back[0][0]):
            between += lines[line_no]
        between += lines[loc_back[0][0]][:loc_back[0][1]]

        return len(between.strip()) == 0

    def merge_removable_triplets(self, removable_triplets, func_triplets):
        def do_merging_triplets():
            nonlocal removable_triplets
            rm_trip_sorted = sorted(removable_triplets, key=lambda l: l[0][0])
            merge = None
            for i in range(len(rm_trip_sorted) - 1):
                if self.check_merge(func_triplets[0], rm_trip_sorted[i][0], rm_trip_sorted[i + 1][0]) and \
                        self.check_merge(func_triplets[1], rm_trip_sorted[i][1], rm_trip_sorted[i + 1][1]) and \
                        self.check_merge(func_triplets[2], rm_trip_sorted[i][2], rm_trip_sorted[i + 1][2]):
                    merge = i
                    break
            if merge is not None:
                removable_triplets = {((min(rm_trip_sorted[merge][0][0], rm_trip_sorted[merge + 1][0][0]),
                                        max(rm_trip_sorted[merge][0][1], rm_trip_sorted[merge + 1][0][1])),
                                       (min(rm_trip_sorted[merge][1][0], rm_trip_sorted[merge + 1][1][0]),
                                        max(rm_trip_sorted[merge][1][1], rm_trip_sorted[merge + 1][1][1])),
                                       (min(rm_trip_sorted[merge][2][0], rm_trip_sorted[merge + 1][2][0]),
                                        max(rm_trip_sorted[merge][2][1], rm_trip_sorted[merge + 1][2][1])))}
                removable_triplets.update({rm_trip_sorted[i] for i in range(len(rm_trip_sorted))
                                           if i != merge and i != merge + 1})
                do_merging_triplets()

        do_merging_triplets()
        return removable_triplets

    @staticmethod
    def do_removing(func: str, removable: Set) -> tuple[str, dict[Any, str]]:
        fcu = FunctionCompareUtilities()
        removed_pieces = dict()
        removable_sorted = sorted(removable, key=lambda l: l[0][0], reverse=True)

        lines = func.splitlines(keepends=True)
        for loc, index in removable_sorted:
            removed_pieces[index] = fcu.retrieve_string_slice(func, loc[0], loc[1])
            if loc[0][0] != loc[1][0]:
                for line_no in range(loc[0][0] + 1, loc[1][0]):
                    lines[line_no] = "\n"
                lines[loc[0][0]] = lines[loc[0][0]][:loc[0][1]] + f"/* placeholder_{index} */" + lines[loc[1][0]][
                                                                                                 loc[1][1]:]
                lines[loc[1][0]] = "\n"
            else:
                lines[loc[0][0]] = lines[loc[0][0]][:loc[0][1]] + f"/* placeholder_{index} */" + lines[loc[1][0]][
                                                                                                 loc[1][1]:]
        return "".join(lines), removed_pieces

    def do_reducing(self, func_before: str, func_after: str, func_mapped: str) -> Dict:
        fcu = FunctionCompareUtilities()

        critical_lines_dict = self.cal_critical_lines(func_before, func_after)
        diff = critical_lines_dict["diff"]
        del_set = critical_lines_dict["del_set"]
        add_set = critical_lines_dict["add_set"]
        
        del_set = {line - 1 for line in del_set}
        add_set = {line - 1 for line in add_set}

        removable_tree_tuples = self.get_removable_tree_tuples(func_before, func_after, del_set, add_set)

        removable_tree_triplets = self.get_removable_tree_triplets(self.remove_preprocs(func_before),
                                                                   self.remove_preprocs(func_mapped),
                                                                   removable_tree_tuples)

        removable_triplets = [(triplet[0]["location"], triplet[1]["location"], triplet[2]["location"])
                              for triplet in removable_tree_triplets]

        removable_triplets = self.merge_removable_triplets(removable_triplets, (func_before, func_after, func_mapped))

        removable_triplets = {(rm[0], rm[1], rm[2], index) for index, rm in enumerate(sorted(removable_triplets,
                                                                                             key=lambda rm: rm[0][0]))}

        func_before_reduced, removed_pieces_before = self.do_removing(func_before,
                                                                     {(rm[0], rm[3]) for rm in removable_triplets})
        func_after_reduced, removed_pieces_after = self.do_removing(func_after,
                                                                   {(rm[1], rm[3]) for rm in removable_triplets})
        func_map_reduced, removed_pieces_mapped = self.do_removing(func_mapped,
                                                                      {(rm[2], rm[3]) for rm in removable_triplets})

        func_before_reduced = fcu.remove_empty_lines(func_before_reduced)
        func_after_reduced = fcu.remove_empty_lines(func_after_reduced)
        func_map_reduced = fcu.remove_empty_lines(func_map_reduced)

        return {
            "upstream_diff": diff,
            "func_before_reduced": func_before_reduced,
            "func_after_reduced": func_after_reduced,
            "func_mapped_reduced": func_map_reduced,
            "removed_pieces_before": removed_pieces_before,
            "removed_pieces_mapped": removed_pieces_mapped
        }

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
