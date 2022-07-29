This directory is different than coref_model_output in the respect that the gold standard annotations that are used to generate the comparisons have been updated.

Files that start with "coref_" are generated using the following comparison schemes:
"_4.txt": section 4 output from hanlp vs. section 4 output in gold standard
"_34.txt": section 3 & 4 output from hanlp vs. section 3 & 4 output in gold standard
"_234.txt": section 2, 3 & 4 output from hanlp vs. section 2, 3 & 4 output in gold standard
"_1234.txt": section 1, 2, 3 & 4 output from hanlp vs. section 1, 2, 3 & 4 output in gold standard

Files that start with "corefon4_" are generated using the following comparison schemes:
"_4.txt": section 4 output from hanlp when only feeding in section 4 vs. section 4 output in gold standard
"_34.txt": section 4 output from hanlp when feeding in section 3 & 4 vs. section 4 output in gold standard
"_234.txt": section 4 output from hanlp when feeding in section 2, 3 & 4 vs. section 4 output in gold standard
"_1234.txt": section 4 output from hanlp when feeding in section 1, 2, 3 & 4 vs. section 4 output in gold standard
