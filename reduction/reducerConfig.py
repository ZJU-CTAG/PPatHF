# change this to your tree-sitter path
tree_sitter_project_path = "tree-sitter-c/"
tree_sitter_lib_path = tree_sitter_project_path + "/build/tree-sitter-c-lib.so"

enough_alpha_threshold = 5

reducing_log_path = ""
reducing_output_path = ""
reducing_input_file = ""

basic_unit_type = {
    # block units
    "if_statement",
    "else_clause",
    "switch_statement",
    "case_statement",
    "for_statement",
    "while_statement",
    "do_statement"
}

levenshtein_similarity_threshold_weak = 0.5
levenshtein_similarity_threshold_parent = 0.55
levenshtein_similarity_threshold_ancestors = 0.45
length_ratio_threshold = 2.5

assert length_ratio_threshold >= 1, "length_ratio_threshold must be greater than or equal to 1!"