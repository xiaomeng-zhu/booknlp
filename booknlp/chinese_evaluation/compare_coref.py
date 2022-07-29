import pandas as pd
import json
import opencc
import os
from model_test import hanlp_coref
from coref_match import csv_to_json

def compare(df1, df2):
    """
    output entries that are in one dataframe but not in the other
    """
    df1 = df1[["start_idx", "end_idx"]]
    df1 = df1.to_records(index=False)
    df1 = list(df1)
    # print(df1)
    df2 = df2[["start_idx", "end_idx"]]
    df2 = df2.to_records(index=False)
    df2 = list(df2)

    in_1_not_in_2 = []
    for trip in df1:
        if trip not in df2:
            in_1_not_in_2.append(trip)

    #print(in_1_not_in_2)

    in_2_not_in_1 = []
    for trip in df2:
        if trip not in df1:
            in_2_not_in_1.append(trip)
    
    print(in_2_not_in_1)

def split_section_to_end(file_path, start_idx):
    """
    given a start character index, return the slice of the text from the given chracter index to the very end
    """
    with open(file_path, "r") as f:
        texts = f.read()
    char_idx = 0
    # need to enumerate all characters in the string because the coreference annotation indices only count characters and punctuations
    for l_idx, l in enumerate(texts):
        if l.isspace() or l == "　":
            pass
        else:
            if char_idx == start_idx:
                return texts[l_idx:]
            char_idx += 1
    return ""

def get_sections(file_path, section_idx_list):
    # useful for getting sections of 4, 34, 234, 1234
    sections = []
    for idx in section_idx_list:
        section = split_section_to_end(file_path, idx)
        sections.append(section)
    return sections

def process_sections_to_json(section_list, text_name, idx_list):
    counts = "1234"
    json_paths = []

    for i in range(len(section_list)):
        # current code is used to generate corefon4
        # last argument gets the first index of the indices list 
        # because we are only interested in hanlp output on the last section when doing "corefon4"
        coref_dict = hanlp_coref(section_list[i], idx_list[i], idx_list[0]) 
        output_name = "annotation/hanlp_corefon4_{}_{}.json".format(text_name, counts[3-i:])

        # for just "coref", use the following two lines instead:
        # coref_dict = hanlp_coref(section_list[i], idx_list[i], idx_list[i]) 
        # output_name = "annotation/hanlp_coref_{}_{}.json".format(text_name, counts[3-i:])

        json_paths.append(output_name)

        with open(output_name, "w") as outfile:
            json.dump(coref_dict, outfile)

        # count += 1
    return json_paths

def compare_to_standard(json_paths, std_json, text_name):
    # using os.system to call scorch does not really work, need to type into console
    for idx, path in enumerate(json_paths):
        suffix = "_scores_"+str(idx+1)+".txt"
        os.system("scorch {} {} {}".format(path, std_json, text_name+suffix))

if __name__ == "__main__":
    # fs_m = pd.read_csv("annotation/miranda_coref_fs.csv")
    # hly_m = pd.read_csv("annotation/miranda_coref_hly.csv")
    # fs_k = pd.read_csv("annotation/kiara_coref_fs.csv")
    # hly_k = pd.read_csv("annotation/kiara_coref_hly.csv")

    fs_standards_csv = ["annotation/standard_coref_fs_4.csv", "annotation/standard_coref_fs_34.csv",
                        "annotation/standard_coref_fs_234.csv", "annotation/standard_coref_fs_1234.csv"]
    for path in fs_standards_csv:
        std = pd.read_csv(path)
        std_json = csv_to_json(std)
        with open(path[:-3] + "json", "w") as outfile:
            json.dump(std_json, outfile)

    hly_standards_csv = ["annotation/standard_coref_hly_4.csv", "annotation/standard_coref_hly_34.csv",
                        "annotation/standard_coref_hly_234.csv", "annotation/standard_coref_hly_1234.csv"]
    for path in hly_standards_csv:
        std = pd.read_csv(path)
        std_json = csv_to_json(std)
        with open(path[:-3] + "json", "w") as outfile:
            json.dump(std_json, outfile)

    fs_section_idx = [1806, 1252, 646, 0]
    hly_section_idx = [2296, 1380, 663, 0]

    # print(compare(fs_m, fs_k))
    # [(569, 569), (689, 691), (936, 937), (1265, 1265), (1625, 1625), (1705, 1706), (1810, 1810), (2160, 2162), (257, 259), (681, 682), (687, 687), (700, 701), (764, 765), (2102, 2103), (1023, 1025), (1950, 1952), (2020, 2022), (2124, 2126), (1034, 1038), (1679, 1680)]
    # [(1034, 1036), (38, 39), (254, 259), (738, 742), (677, 682), (689, 695), (697, 701), (759, 765), (2097, 2103), (1923, 1925), (2334, 2338), (2160, 2164), (2132, 2134)]

    # print(compare(hly_m, hly_k))
    # [(186, 187), (1327, 1330), (2003, 2004), (2321, 2323), (597, 598), (1767, 1767), (1393, 1396), (1807, 1813), (1045, 1047), (847, 850), (893, 894), (899, 902), (1092, 1092), (1104, 1106), (1205, 1205), (2164, 2165), (2325, 2326), (981, 982), (987, 988), (1401, 1404), (1833, 1834)]
    # [(3, 5), (105, 106), (11, 13), (1329, 1330), (1364, 1364), (1807, 1809), (2002, 2003), (2320, 2323), (17, 18), (843, 850), (898, 902), (1103, 1106), (2567, 2570), (2636, 2642), (161, 162), (446, 447), (1675, 1677), (238, 239), (1392, 1396), (1810, 1813), (1077, 1079), (917, 924), (1403, 1404), (1828, 1834)]

    # fs_sections = get_sections("examples/fengshou_chapter1.txt", fs_section_idx)
    # hly_sections = get_sections("examples/with_poetry/huliyuanquanzhuan_chapter1.txt", hly_section_idx)

    # converter = opencc.OpenCC('t2s.json')
    # fs_sections_simp = [converter.convert(section) for section in fs_sections]
    # hly_sections_simp = [converter.convert(section) for section in hly_sections]

    # fs_json_paths = process_sections_to_json(fs_sections_simp, "fs", fs_section_idx)
    # hly_json_paths = process_sections_to_json(hly_sections_simp, "hly", hly_section_idx)

    # fs_std_json = ["annotation/standard_coref_fs_4.json", "annotation/standard_coref_fs_34.json",
    #             "annotation/standard_coref_fs_234.json", "annotation/standard_coref_fs_1234.json"]

    # compare_to_standard(fs_json_paths, fs_std_json, "fs")
