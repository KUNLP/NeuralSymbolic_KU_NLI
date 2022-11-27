import json
from tqdm import tqdm
from operator import itemgetter

def change_tag(change_dic, tag_list, text):
    #print("change_dic: " + str(change_dic))
    del_dic = []
    for key in change_dic.keys():
        if change_dic[key][0] in change_dic.keys():
            del_dic.append(change_dic[key][0])
            change_dic[key] = change_dic[change_dic[key][0]]
            #print(del_dic)
    for dic in set(del_dic):
        del change_dic[dic]

    dic_val_idx = [val[1] for val in change_dic.values()]
    for i, val_idx in enumerate(dic_val_idx):
        for j, val2 in zip([j for j, val2 in enumerate(dic_val_idx) if j != i], [val2 for j, val2 in enumerate(dic_val_idx) if j != i]):
            #print(str(i) + " " + str(j))
            #print(val_idx)
            #print(val2)
            if (len(set(val_idx).intersection(set(val2))) != 0) or (len(set(val2).intersection(set(val_idx))) != 0):
                #print(str(i)+" "+str(j))
                change_dic[list(change_dic.keys())[i]] = [" ".join([text.split()[k] for k in list(set(
                    change_dic[list(change_dic.keys())[i]][1] + change_dic[list(change_dic.keys())[j]][1]))]),
                list(set(change_dic[list(change_dic.keys())[i]][1] + change_dic[list(change_dic.keys())[j]][1]))]

                change_dic[list(change_dic.keys())[j]] = [" ".join([text.split()[k] for k in list(set(
                    change_dic[list(change_dic.keys())[i]][1] + change_dic[list(change_dic.keys())[j]][1]))]),
                                                          list(set(change_dic[list(change_dic.keys())[i]][1] +
                                                                   change_dic[list(change_dic.keys())[j]][1]))]
                #print("change_dic: " + str(change_dic))
                dic_val_idx = [val[1] for val in change_dic.values()]
    #print("change_dic: " + str(change_dic))
    for tag_idx, tag_li in enumerate(tag_list):
        del_list = []
        for tag_i,tag_l in enumerate(tag_li):
            if tag_l[0][0] in change_dic.keys():
                tag_list[tag_idx][tag_i][2][0] = change_dic[tag_l[0][0]][1]
                tag_list[tag_idx][tag_i][0][0] = change_dic[tag_l[0][0]][0]
            if tag_l[0][1] in change_dic.keys():
                tag_list[tag_idx][tag_i][2][1] = change_dic[tag_l[0][1]][1]
                tag_list[tag_idx][tag_i][0][1] = change_dic[tag_l[0][1]][0]

            if tag_l[0][0] == tag_l[0][1]: del_list.append(tag_l)
        tag_list[tag_idx] = [x for x in tag_list[tag_idx] if x not in del_list]

    return tag_list, {}

def merge_tag(inf_dir, outf_dir, tag_li_type = "modifier_w/_phrase"):
    dp = inf_dir.split("_")[0].split("/")[-1]
    with open(inf_dir, "r", encoding="utf-8") as inf:
        datas = json.load(inf)

    outputs = [];outputs2 = [];
    for id, data in tqdm(enumerate(datas)):
        output = [];output2 = [];
        # {"origin": text, dp:root, words:[[word, tag, idx], [], ... ]}
        texts_list = [data["premise"], data["hypothesis"]]
        #print("\n=================="+str(data["guid"])+"=============================")
        for texts in texts_list:
            #print("\n--------------------------------------------------------------------")
            text = texts["origin"]
            #print(text)
            #print("--------------------------------------------------------------------")

            # [{'R', 'VNP', 'L', 'VP', 'S', 'AP', 'NP', 'DP', 'IP', 'X'}, {'None', 'MOD', 'CNJ', 'AJT', 'OBJ', 'SBJ', 'CMP'}]
            r_list = []; l_list = []; s_list = []; x_list = []; np_list = []; dp_list = []; vp_list = [];vnp_list = []; ap_list = []; ip_list = []
            MOD = []; AJT = []; CMP = [];
            np_cnj_list = []
            #print(texts[dp]["words"])

            # 전처리
            ## [['어떤 방에서도', '금지됩니다.'], ['NP', 'AJT'], [[0, 1], [3]]], [['방에서도', '흡연은'], ['NP', 'MOD'], [[1], [2]]]와 같은 경우
            for i, koala in enumerate(texts[dp]["words"]):
                for j, other in enumerate(texts[dp]["words"]):
                    if (texts[dp]["words"][i][2][0] != other[2][0]) and (set(texts[dp]["words"][i][2][0]+other[2][0]) == set(other[2][0])):
                        texts[dp]["words"][i][0][0] = other[0][0]
                        texts[dp]["words"][i][2][0] = other[2][0]
                    if (texts[dp]["words"][i][2][0] != other[2][1]) and (set(texts[dp]["words"][i][2][0]+other[2][1]) == set(other[2][1])):
                        texts[dp]["words"][i][0][0] = other[0][1]
                        texts[dp]["words"][i][2][0] = other[2][1]
                    if (texts[dp]["words"][i][2][1] != other[2][0]) and (set(texts[dp]["words"][i][2][1]+other[2][0]) == set(other[2][0])):
                        texts[dp]["words"][i][0][1] = other[0][0]
                        texts[dp]["words"][i][2][1] = other[2][0]
                    if (texts[dp]["words"][i][2][1] != other[2][1]) and (set(texts[dp]["words"][i][2][1]+other[2][1]) == set(other[2][1])):
                        texts[dp]["words"][i][0][1] = other[0][1]
                        texts[dp]["words"][i][2][1] = other[2][1]
            #print(texts[dp]["words"])
            for i, koala in enumerate(texts[dp]["words"]):
                #print(koala)
                tag = koala[1]
                word_idx = koala[2]
                if (tag[0] == "NP") and (tag[1] != "CNJ"):
                    np_list.append(koala)
                elif (tag[0] == "DP"):
                    dp_list.append(koala)
                elif (tag[0] == "VP"):
                    vp_list.append(koala)
                elif (tag[0] == "VNP"):
                    vnp_list.append(koala)
                elif (tag[0] == "AP"):
                    ap_list.append(koala)
                elif (tag[0] == "IP"):
                    ip_list.append(koala)
                elif (tag[0] == "R"):
                    r_list.append(koala)
                elif (tag[0] == "L"):
                    l_list.append(koala)
                elif (tag[0] == "S"):
                    s_list.append(koala)
                elif (tag[0] == "X"):
                    x_list.append(koala)

                if  (tag[1] == "MOD"): MOD.append(koala);
                elif (tag[1] == "AJT"): AJT.append(koala);
                elif (tag[1] == "CMP"): CMP.append(koala);


                if (tag[0] == "NP") and (tag[1] == "CNJ"):
                    np_cnj_list.append(koala)

            vp_list = vp_list+vnp_list
            tag_list = []
            if tag_li_type == "modifier":
                tag_list = [MOD+ AJT+ CMP] #3
                # tag_list = [MOD, AJT, CMP] #2
            elif tag_li_type == "phrase":
                tag_list = [x for x in [np_list, dp_list, vp_list, ap_list, ip_list, r_list, l_list, s_list, x_list] if len(x) != 0] # 1
            elif tag_li_type == "modifier_w/_phrase":
                tag_list = [MOD+ AJT+ CMP]+[x for x in [np_list, dp_list, vp_list, ap_list, ip_list, r_list, l_list, s_list, x_list] if len(x) != 0] #4

            change_dic = {}
            for tag_idx,tag_li in enumerate(tag_list):
                #print("tag_li: "+str(tag_li))
                conti = True
                while conti:
                    tag_list, change_dic = change_tag(change_dic, tag_list, text)
                    tag_li = tag_list[tag_idx]
                    #print("tag_li: " + str(tag_li))
                    new_tag_li = []
                    other_tag_li = []
                    for tag_l in tag_li:
                        if abs(max(tag_l[2][1])-min(tag_l[2][0]))==1:new_tag_li.append(tag_l)
                        else: other_tag_li.append(tag_l)

                    # print("new_tag_li: "+str(new_tag_li))

                    if (len(new_tag_li)==0) or (len(tag_li)==1):
                        conti = False;
                    else:
                        new_tag_li = sorted(new_tag_li, key = lambda x:(max(x[2][1])))
                        # print("new_tag_li after sorted: "+str(new_tag_li))

                        tag_li = other_tag_li
                        del other_tag_li
                        #print("tag_li: " + str(tag_li))

                        # 거리의 길이가 1일 때 양 옆에 이어서 있는 경우
                        i = 1
                        while (i != len(new_tag_li)):
                            if (new_tag_li[-i+1][0][1] == new_tag_li[-i][0][0]) and (new_tag_li[-i+1][2][1] == new_tag_li[-i][2][0]):
                                if min(new_tag_li[-i][2][0])<min(new_tag_li[-i][2][1]):
                                    change_dic.update({new_tag_li[-i][0][0]: [" ".join(new_tag_li[-i][0]), list(set(new_tag_li[-i+1][2][1]+new_tag_li[-i][2][0]+new_tag_li[-i][2][1]))]})
                                    change_dic.update({new_tag_li[-i][0][1]: [" ".join(new_tag_li[-i][0]), list(set(new_tag_li[-i + 1][2][1] + new_tag_li[-i][2][0] + new_tag_li[-i][2][1]))]})
                                elif min(new_tag_li[-i][2][0])>min(new_tag_li[-i][2][1]):
                                    change_dic.update({new_tag_li[-i][0][0]: [" ".join([new_tag_li[-i][0][1], new_tag_li[-i][0][0]]), list(set(new_tag_li[-i+1][2][1]+new_tag_li[-i][2][0]+new_tag_li[-i][2][1]))]})
                                    change_dic.update({new_tag_li[-i][0][1]:[" ".join([new_tag_li[-i][0][1], new_tag_li[-i][0][0]]), list(set(new_tag_li[-i + 1][2][1] + new_tag_li[-i][2][0] + new_tag_li[-i][2][1]))]})
                                new_tag_li[-i + 1] = [[new_tag_li[-i+1][0][0], " ".join(new_tag_li[-i][0])], new_tag_li[-i+1][1], [new_tag_li[-i+1][2][0], list(set(new_tag_li[-i+1][2][1]+new_tag_li[-i][2][0]+new_tag_li[-i][2][1]))]]
                                new_tag_li.pop()
                            else: i+=1

                        # 거리의 길이가 2인 경우
                        del_list = []
                        for new_tag_l in new_tag_li:
                            for tag_l in tag_li:
                                if (tag_l[0][1] == new_tag_l[0][1]) and (max(tag_l[2][0])+1 == min(new_tag_l[2][0])):
                                    if min(tag_l[2][0]) < min(new_tag_l[2][0]):
                                        change_dic.update({new_tag_l[0][0]:[" ".join([tag_l[0][0], new_tag_l[0][0]]), list(set(tag_l[2][0] + new_tag_l[2][0]))]})
                                        change_dic.update({tag_l[0][0]:[" ".join([tag_l[0][0], new_tag_l[0][0]]),list(set(tag_l[2][0] + new_tag_l[2][0]))]})
                                    elif min(tag_l[2][0]) > min(new_tag_l[2][0]):
                                        change_dic.update({new_tag_l[0][0]: [" ".join([new_tag_l[0][0], tag_l[0][0]]),
                                                                             list(set(tag_l[2][0] + new_tag_l[2][0]))]})
                                        change_dic.update({tag_l[0][0]: [" ".join([new_tag_l[0][0], tag_l[0][0]]),
                                                                         list(set(tag_l[2][0] + new_tag_l[2][0]))]})

                                    del_list.append(new_tag_l)

                        new_tag_li = [x for x in new_tag_li if x not in del_list]

                        #print("new_tag_li: "+str(new_tag_li))
                        if len(new_tag_li) != 0:
                            for new_tag_l in new_tag_li:
                                if (min(new_tag_l[2][0]) < min(new_tag_l[2][1])) and (max(new_tag_l[2][0])+1 == min(new_tag_l[2][1])):
                                    change_dic.update({new_tag_l[0][0]:[" ".join(new_tag_l[0]), list(set(sum(new_tag_l[2],[])))]})
                                    change_dic.update({new_tag_l[0][1]: [" ".join(new_tag_l[0]), list(set(sum(new_tag_l[2], [])))]})
                                elif (min(new_tag_l[2][0]) > min(new_tag_l[2][1])) and (max(new_tag_l[2][1])+1 == min(new_tag_l[2][0])):
                                    change_dic.update({new_tag_l[0][0]:[" ".join([new_tag_l[0][1], new_tag_l[0][0]]), list(set(sum(new_tag_l[2],[])))]})
                                    change_dic.update({new_tag_l[0][1]: [" ".join([new_tag_l[0][1], new_tag_l[0][0]]), list(set(sum(new_tag_l[2], [])))]})
                            new_tag_li = []

                        if change_dic == {}:
                            conti = False;
                        #print("change_dic: " + str(change_dic))

                tag_list[tag_idx] = tag_li
                #print("change tag_li: " + str(tag_list[tag_idx]))
            #print("tag_list: "+ str(tag_list))
            tag_list = [x for x in tag_list if x != []]

            #for i, koala in enumerate(texts[dp]["words"]):
            #    # print(koala)
            #    tag = koala[1]
            #    if (tag[0] == "NP") and (tag[1] == "CNJ"):
            #        if (koala[2][0] != koala[2][1]):np_cnj_list.append(koala)
            #"""
            new_koalas = [[] for tag_idx, _ in enumerate(tag_list)]
            for tag_idx, _ in enumerate(tag_list):
                # NP-CNJ
                if (len(np_cnj_list) != 0):
                    for cnj_koala in np_cnj_list:
                        for ttag_list in tag_list[tag_idx]:
                            new_koala = []
                            if (len(set(cnj_koala[2][0]).intersection(set(ttag_list[2][0]))) != 0):
                                new_koala = [[cnj_koala[0][0], ttag_list[0][1]],
                                             cnj_koala[1],
                                             [cnj_koala[2][0], ttag_list[2][1]]]

                            if (len(set(cnj_koala[2][0]).intersection(set(ttag_list[2][1]))) != 0):
                                new_koala = [[ttag_list[0][1], cnj_koala[0][1]],
                                             cnj_koala[1],
                                             [ttag_list[2][1], cnj_koala[2][1]]]

                            if (len(set(cnj_koala[2][1]).intersection(set(ttag_list[2][0]))) != 0):
                                new_koala = [[cnj_koala[0][0], ttag_list[0][0]],
                                             cnj_koala[1],
                                             [cnj_koala[2][0], ttag_list[2][0]]]

                            if (len(set(cnj_koala[2][1]).intersection(set(ttag_list[2][1]))) != 0):
                                new_koala = [[cnj_koala[0][0], ttag_list[0][1]],
                                             cnj_koala[1],
                                             [cnj_koala[2][0], ttag_list[2][1]]]

                            if new_koala != []: new_koalas[tag_idx].append(new_koala)

            for k_idx, new_koala in enumerate(new_koalas):
                new_tag_list = tag_list[k_idx]+new_koala
                tag_list[k_idx] = new_tag_list
            #"""

            tag_list = sum(tag_list, [])
            tag_list = [tag for tag in tag_list if len(set(tag[2][0]).intersection(set(tag[2][1]))) == 0]
            #print("tag_list: " + str(tag_list))

            sub_output2 = {}
            for tag in tag_list:
                #print(tag)
                sub_output2[tag[0][0]] = tag[2][0]
                sub_output2[tag[0][1]] = tag[2][1]
            sub_output2 = [[key, sorted(value)] for key,value in sub_output2.items()]
            # print(sub_output2)
            sub_output2.sort(key=itemgetter(1))
            # print("sub_output2: " + str(sub_output2))
            #  후처리
            i = 0
            while (i < len(sub_output2) - 1):
                sub1 = sub_output2[i]
                sub2 = sub_output2[i + 1]
                if (sub1[0] != sub2[0]) and (sub1[1] != sub2[1]) and ((len(set(sub1[1]).intersection(set(sub2[1]))) != 0) or (len(set(sub2[1]).intersection(set(sub1[1]))) != 0)):
                    #print("sub_output2: " + str(sub_output2))
                    new_idx = sorted(list(set(sub1[1] + sub2[1])))
                    sub_output2[i] = (" ".join(texts["origin"].split()[min(new_idx):max(new_idx) + 1]), new_idx)
                    sub_output2 = sub_output2[:i + 1] + sub_output2[i + 2:]
                    #print("sub_output2: " + str(sub_output2))
                    for j, tag in enumerate(tag_list):
                        if (tag[0][0] in [sub1[0], sub2[0]]):
                            tag_list[j][0][0] = " ".join(texts["origin"].split()[min(new_idx):max(new_idx) + 1])
                            tag_list[j][2][0] = new_idx
                        elif (tag[0][1] in [sub1[0], sub2[0]]):
                            tag_list[j][0][1] = " ".join(texts["origin"].split()[min(new_idx):max(new_idx) + 1])
                            tag_list[j][2][1] = new_idx
                else: i += 1

            # print("sub_output2: " + str(sub_output2))
            # print("tag_list: " + str(tag_list))

            if sub_output2 == []:
                sub_output2 = [[" ".join(text.split()[:-1]), [i for i in range(0, len(text.split())-1)]], [text.split()[-1], [len(text.split())-1]]]
                tag_list = [[[sub_output2[0][0], sub_output2[1][0]], ['VP', 'MOD'], [sub_output2[0][1], sub_output2[1][1]]]]

            if sum([sub[1] for sub in sub_output2], []) != [i for i,_ in enumerate(text.split())]:
                for li in sum([[sorted(tag[2][0]), sorted(tag[2][1])] for tag in tag_list], []):
                    if (len(set(li).intersection(set([t for t, _ in enumerate(text.split()) if t not in sum([sub[1] for sub in sub_output2], [])]))) != 0):
                        sub_output2 += [[" ".join([text.split()[t] for t in li]), li]]
                sub_output2 += [[t, [i]] for i, t in enumerate(text.split()) if i not in sum([sub[1] for sub in sub_output2], [])]
            sub_output2.sort(key=itemgetter(1))
            #print("sub_output2: " + str(sub_output2))
            #print("tag_list: " + str(tag_list))
            #print(sum([sub[1] for sub in sub_output2], []))
            #print([[i, t] for i, t in enumerate(text.split())])
            assert sum([sub[1] for sub in sub_output2], []) == [i for i,_ in enumerate(text.split())]

            sub_output2 = [[sub[0], sorted(sub[1])] for sub in sub_output2]
            output2.append(sub_output2)
            tag_list = [[tag[0], tag[1], [sorted(tag[2][0]), sorted(tag[2][1])]] for tag in tag_list]
            #tag_list += np_cnj_list
            output.append(tag_list)
        #print("output2: " + str(output2))
        #print("output: " + str(output))
        outputs.append(output)
        # sub_output2: [('흡연자분들은', [0]), ('발코니가 있는', [1, 2]), ('방이면', [3]), ('발코니에서 흡연이', [4, 5]), ('가능합니다.', [6])]
        outputs2.append(output2)

    for i, (data, merge1, merge2) in enumerate(zip(datas, outputs, outputs2)):
        datas[i]["premise"]["merge"] = {"origin": merge2[0], dp: merge1[0]}
        datas[i]["hypothesis"]["merge"] = {"origin": merge2[1], dp: merge1[1]}

    for i, data in tqdm(enumerate(datas)):
        for sen in ["premise", "hypothesis"]:
            for j, merge in enumerate(data[sen]["merge"]["origin"]):
                if merge == ["", []]:
                    data[sen]["merge"]["origin"] = [data[sen]["merge"]["origin"][j+1]]
                    datas[i][sen]["merge"]["origin"] = data[sen]["merge"]["origin"]
                    data[sen]["merge"]["parsing"][0][0][0] = data[sen]["merge"]["parsing"][0][0][1]
                    data[sen]["merge"]["parsing"][0][2][0] = data[sen]["merge"]["parsing"][0][2][1]
                    datas[i][sen]["merge"]["parsing"] = data[sen]["merge"]["parsing"]

        for sen in ["premise", "hypothesis"]:
            new_koala = []
            for k, words in enumerate(data[sen]["merge"][dp]):
                origin_idx = [merge[1] for merge in data[sen]["merge"]["origin"]]
                if (words[2][0] in origin_idx):
                    if (words[2][1] in origin_idx):
                        new_koala.append(words)
            datas[i][sen]["merge"][dp] = new_koala


    with open(outf_dir, 'w', encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)
    print("\n\nfinish!!")

if __name__ =='__main__':

    inf_dirs = ["./../../data/parsing/parsing_1_klue_nli_train.json", "./../../data/parsing/parsing_1_klue_nli_dev.json"]
                #["./../../data/koala/koala_ver1_klue_nli_train.json", "./../../data/koala/koala_ver1_klue_nli_dev.json"]
    outf_dirs = ["./../../data/merge/parsing_1_klue_nli_train.json", "./../../data/merge/parsing_1_klue_nli_dev.json"]
                #["./../../data/merge/merge_3_klue_nli_train.json", "./../../data/merge/merge_3_klue_nli_dev.json"]

    for inf_dir, outf_dir in zip(inf_dirs, outf_dirs):
        merge_tag(inf_dir, outf_dir, tag_li_type = "modifier_w/_phrase")
        # merge_tag(inf_dir, outf_dir, tag_li_type = "modifier")
        # merge_tag(inf_dir, outf_dir, tag_li_type = "phrase")

