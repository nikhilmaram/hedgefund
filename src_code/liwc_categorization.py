import string
import pandas as pd
from typing import List
from typing import Tuple
import os
import multiprocessing
import time

import config as cfg
import misc
import employee

CAT_DELIM = "%"


pd.set_option('display.max_colwidth', -1)
# =========================================================================
# ============= Generating LIWC parser ====================================
# =========================================================================
def parse_liwc(file_path:str,whitelist=None):
    """Parsers LIWC dictionary and returns a dictionary.

    Args:
        file_path : path to the dictionary file.

    Returns:
        words_to_cats : Word to category dictionary.

    """
    f = open(file_path)
    cats_section = False
    words_to_cats = {}
    id_to_cat = {}
    weird_lines = {}
    for l in f:
        l = l.strip()
        if l == CAT_DELIM:
            cats_section = not cats_section
            continue

        if cats_section:
            try:
                i, cat = l.split("\t")
                cat = cat.split()[0]
                id_to_cat[int(i)] = cat
            except: pass # likely hierarchical category tags
        else:
            w, cats = l.split("\t")[0], l.split("\t")[1:]
            if "(" in w and ")" in w:
                w = w.replace("(","").replace(")","")
            try:
                words_to_cats[w] = [id_to_cat[int(i)] for i in cats]
            except:
                weird_lines[w] = cats

    # Finetuning cause like is weird
    discrep = [w for w,cs in words_to_cats.items() if id_to_cat[53] in cs]
    cs = words_to_cats["53 like*"]
    words_to_cats.update({d+" like*": cs for d in discrep})
    del words_to_cats["53 like*"]

    ## If whitelist
    if whitelist:
        words_to_cats = {w: [c for c in cs if c in whitelist] for w,cs in words_to_cats.items()}
        words_to_cats = {w:cs for w,cs in words_to_cats.items() if cs}

    return words_to_cats

def preprocess(text):
    """ Preprocess the input string to return list of words.

    Args:
        text : string.
    Returns:
        tokens_list: list of words present in string.
        l : number of words

    Tokenizes, removes trailing punctuation from words, counts how many words"""

    text = text.lower().replace("kind of", "kindof")
    def strip_punct(x):
        if all([c in string.punctuation for c in x]):
            return x
        else:
            return x.strip(string.punctuation)

    tokens_list = [strip_punct(w) for w in text.split()]
    l = len(tokens_list)

    return tokens_list, l

def _extract(lex, tokens_list, n_words, percentage=True, wildcard="*"):
    """Returns category dictionary based on input tokens list.

    Args:
        lex             : LIWC dictionary.
        tokens_list     : tokens list after preprocessing the input text.
        n_words         : number of words.
        percentage      : Percentage of categories is returned.

    Returns:
        extracted       : dict, key - category, value - percentage of words in the category.


    """
    extracted = {}
    is_weighted = isinstance(list(lex.items())[0][1], dict)

    if wildcard == "":
        wildcard = "~$¬"  # highly unlikely combo

    for w, cats in lex.items():
        w_split = w.split()
        # split -> bigram expression
        if not any([t.replace(wildcard, "") in " ".join(tokens_list) for t in w_split]):
            continue

        if wildcard in w:
            ngrams = [[t.startswith(w_t.replace(wildcard, "")) for t, w_t in zip(tp, w_split)]
                      for tp in zip(*[
                    tokens_list[i:] for i in range(len(w_split))])]
            count = sum(map(all, ngrams))
        else:
            ngrams = [list(t) for t in zip(*[
                tokens_list[i:] for i in range(len(w_split))])]
            count = ngrams.count(w_split)

        if count:
            for c in cats:
                if is_weighted:
                    wg = cats[c]
                else:
                    wg = 1
                extracted[c] = extracted.get(c, 0) + (count * wg)

    if percentage:
        ## Turn into percentages
        extracted = {k: v / n_words for k, v in extracted.items()}
    return extracted

def extract(lex,doc,percentage=True,wildcard="*"):
    """
    Counts all categories present in the document given the lexicon dictionary.
    percentage (optional) indicates whether to return raw counts or
    normalize by total number of words in the document
      Args:
        lex             : LIWC dictionary.
        tokens_list     : tokens list after preprocessing the input text.
        n_words         : number of words.
        percentage      : Percentage of categories is returned.

    Returns:
        extracted       : dict, key - category, value - percentage of words in the category.

    """

    tokens_list, n_words = preprocess(doc)
    return _extract(lex,tokens_list,n_words,percentage,wildcard=wildcard)

def reverse_dict(d):
    cats_to_words = {}
    for w, cs in d.items():
        for c in cs:
            l = cats_to_words.get(c, set())
            l.add(w)
            cats_to_words[c] = l
    return cats_to_words

# =========================================================================
# ==================== Compute necessary dictionaries =====================
# =========================================================================

lex_dict = parse_liwc(cfg.LIWC_DICTIONARY)

# =========================================================================
# ==================Get required category values from the text=============
# =========================================================================

def compute_required_categories_from_text(text : str) -> dict:
    """compute required categories from the text.

    Args:
        text : input text whose LIWC dictionary needs to be computed.

    Returns:
        category_dict : category dict values.
    """
    counter_dict = extract(lex_dict, text, percentage=True)
    category_dict = {}

    category_dict["cognitive_process"] = counter_dict.get("cogproc",0)
    category_dict["insight"] = counter_dict.get("insight",0)
    category_dict["causation"] = counter_dict.get("cause",0)
    category_dict["certainity"] = counter_dict.get("certain", 0)
    category_dict["discrepancy"] = counter_dict.get("discrep", 0)
    category_dict["tentativeness"] = counter_dict.get("tentat", 0)
    category_dict["differentiation"] = counter_dict.get("differ", 0) ## same as exlcusion.

    category_dict["affect_process"] = counter_dict.get("affect", 0)
    category_dict["positive_emotion"] = counter_dict.get("posemo", 0)
    category_dict["negative_emotion"] = counter_dict.get("negemo", 0)
    category_dict["anxiety"] = counter_dict.get("anx", 0)
    category_dict["anger"] = counter_dict.get("anger", 0)
    category_dict["sadness"] = counter_dict.get("sad", 0)


    return category_dict


# =========================================================================
# =================compute LIWC categories of user messages==============
# =========================================================================

def compute_liwc_user_messages(im_df : pd.DataFrame, user_name: str, complete_network:bool = False, in_network:bool = True) -> Tuple[dict,dict]:
    """compute LIWC categories of user messages

    Args:
        im_df               : Input Dataframe.
        user_name           : user name
        complete_network    : complete network to be considered (True: complete network, False: network depending on in_network).
        in_network          : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: outside)

    Returns:
        sent_messages_category_dict       : LIWC categories of sent messages from the user.
        recv_messages_category_dict       : LIWC categories of received messages to the user.
    """

    im_df_sent = im_df[im_df["sender_user_name"] == user_name]
    if not complete_network:
        if in_network:
            im_df_sent = im_df_sent[im_df_sent["receiver_in_network"] == 1]
        else:
            im_df_sent = im_df_sent[im_df_sent["receiver_in_network"] == 0]

    ## dataframe with receiver as input user.
    im_df_recv = im_df[im_df["receiver_user_name"] == user_name]
    if not complete_network:
        if in_network:
            im_df_recv = im_df_recv[im_df_recv["sender_in_network"] == 1]
        else:
            im_df_recv = im_df_recv[im_df_recv["sender_in_network"] == 0]


    sent_message_list = im_df_sent["content"].tolist()
    recv_message_list = im_df_recv["content"].tolist()

    sent_messages = " ".join(sent_message_list)
    recv_messages = " ".join(recv_message_list)

    sent_messages_category_dict = compute_required_categories_from_text(sent_messages)
    recv_messages_category_dict = compute_required_categories_from_text(recv_messages)

    return sent_messages_category_dict,recv_messages_category_dict

# =========================================================================
# =================compute LIWC categories of messages of user list======
# =========================================================================

def compute_liwc_user_list_messages(im_df: pd.DataFrame, user_name_list: List, complete_network: bool = False,
                         in_network: bool = True) -> Tuple[dict, dict,dict]:
    """compute LIWC of user list messages.

    Args:
        im_df               : Input Dataframe.
        user_name_list      : user name list.
        complete_network    : complete network to be considered (True: complete network, False: network depending on in_network).
        in_network          : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: outside)

    Returns:
        sent_messages_category_dict       : LIWC categories of sent messages from the user list.
        recv_messages_category_dict       : LIWC categories of received messages to the user list.
        within_messages_category_dict     : LIWC categories of messages within the user list.
    """

    im_df_sent = im_df[im_df["sender_user_name"].isin(user_name_list)]
    ## This might also contains receivers which are in user_name_list, so removing the rows in which receivers are not in user_name_list.
    im_df_sent = im_df_sent[~im_df_sent["receiver_user_name"].isin(user_name_list)]
    if not complete_network:
        if in_network:
            im_df_sent = im_df_sent[im_df_sent["receiver_in_network"] == 1]
        else:
            im_df_sent = im_df_sent[im_df_sent["receiver_in_network"] == 0]

    im_df_recv = im_df[im_df["receiver_user_name"].isin(user_name_list)]
    im_df_recv = im_df_recv[~im_df_recv["sender_user_name"].isin(user_name_list)]
    if not complete_network:
        if in_network:
            im_df_recv = im_df_recv[im_df_recv["sender_in_network"] == 1]
        else:
            im_df_recv = im_df_recv[im_df_recv["sender_in_network"] == 0]

    im_df_within = im_df[im_df["sender_user_name"].isin(user_name_list)]
    im_df_within = im_df_within[im_df_within["receiver_user_name"].isin(user_name_list)]

    sent_message_list = im_df_sent["content"].tolist()
    recv_message_list = im_df_recv["content"].tolist()
    within_message_list = im_df_within["content"].tolist()

    sent_message_list = [x.replace('"','') for x in sent_message_list]
    recv_message_list = [x.replace('"', '') for x in recv_message_list]
    within_message_list = [x.replace('"', '') for x in within_message_list]


    sent_messages = ' '.join(sent_message_list)
    recv_messages = " ".join(recv_message_list)
    within_messages = " ".join(within_message_list)

    start_time  = time.time()
    sent_messages_category_dict = compute_required_categories_from_text(sent_messages)
    recv_messages_category_dict = compute_required_categories_from_text(recv_messages)
    within_messages_category_dict = compute_required_categories_from_text(within_messages)
    # print("Time elapsed for computing all category dicts : {0}".format(time.time()-start_time))

    del [im_df_sent, im_df_recv, im_df_within]

    return sent_messages_category_dict, recv_messages_category_dict, within_messages_category_dict


# =========================================================================
# =================Compute category dict given file list.==================
# =========================================================================

def compute_liwc_categories_from_filelist(file_list : List, src_dir_path:str, user_name_list:List,
                                     return_dict:dict, complete_network:bool= False,in_network:bool = True,only_week:bool=False):
    """Computes categories from all files in file_list for the users present in user_name_list.
     Args:
         src_dir_path : path to the input files.
         user_name_list : user names for whose sentiment needs to be computed.
         return_dict : To pass the return value for multiprocess invocation.
         complete_network : complete network to be considered (True: complete network, False: network depending on in_network).
         in_network : Boolean Variable determines whether the messages within the hedgefund network be considered. (True : In , False: outside)
         only_week           : data is computed weekly instead of each date.

     total_sent_category_dict = liwc category dictionary of sent messages. key - date ; value - category dict of send messages on the date.
     total_recv_category_dict = liwc category dictionary of received messages. key - date ; value - category dict of received messages on the date.
     total_within_category_dict = liwc category dictionary of within messages. key - date ; value - category dict of within messages on the date.
     """
    process_count = multiprocessing.current_process().name

    print("In compute_categories_from_filelist function", only_week)

    total_sent_category_dict = {}
    total_recv_category_dict = {}
    total_within_category_dict = {}

    for file_name in file_list:
        # print(file_name)
        file_path = os.path.join(src_dir_path,file_name)
        df = pd.read_csv(file_path)

        if only_week:
            ## Category dict is computed over week data.
            sent_category_dict, recv_category_dict, within_category_dict = compute_liwc_user_list_messages(
                df, user_name_list,complete_network=complete_network,in_network=in_network)
            ## Calculate the monday of the week.
            week_num = int(file_name.split('.')[0].split('_')[-1][4:])
            curr_date = misc.calculate_date(week_num=week_num).strftime("%Y-%m-%d")
            total_sent_category_dict[curr_date] = sent_category_dict
            total_recv_category_dict[curr_date] = recv_category_dict
            total_within_category_dict[curr_date] = within_category_dict
        else:
            ## Group the data according to curr_date.
            for curr_date, df_curr_date in df.groupby("day"):
                sent_category_dict, recv_category_dict, within_category_dict = compute_liwc_user_list_messages(
                    df_curr_date,user_name_list, complete_network=complete_network,in_network=in_network)
                total_sent_category_dict[curr_date] = sent_category_dict
                total_recv_category_dict[curr_date] = recv_category_dict
                total_within_category_dict[curr_date] = within_category_dict

    return_dict[process_count] = [total_sent_category_dict, total_recv_category_dict, total_within_category_dict]
    # return sent_sentiment_dict, recv_sentiment_dict, within_sentiment_dict



# =========================================================================================
# =================Compute category dict given file list.(Multi- Process)==================
# =========================================================================================

def compute_liwc_categories_from_filelist_multiproc(src_dir_path : str, user_name_list: List,num_process :int,
                                               complete_network: bool = False, in_network:bool = True, start_week : int = 0,
                                               end_week : int = 264,only_week:bool=False):
    """Runs the compute_liwc_categories_from_filelist on multiple process.

    Args:
        src_dir_path        : path to the files present in file_list
        user_name_list      : users for whom sentiment is calculated.
        num_process         : Number of process.
        complete_network    : complete network to be considered (True: complete network, False: network depending on in_network).
        in_network          : messages considered within/outside network.
        start_week          : starting week of files to be considered.
        end_week            : Ending week of the to be considered.
        only_week           : data is calculated weekly instead of each date.

    Returns:
        total_sent_category_dict = liwc category dictionary of sent messages. key - date ; value - category dict of send messages on the date.
        total_recv_category_dict = liwc category dictionary of received messages. key - date ; value - category dict of received messages on the date.
        total_within_category_dict = liwc category dictionary of within messages. key - date ; value - category dict of within messages on the date.
    """

    total_sent_category_dict = {}
    total_recv_category_dict = {}
    total_within_category_dict = {}
    list_of_file_list = misc.splitting_all_files(src_dir_path, num_process,start_week,end_week)

    process_list = []

    ## Need a dictionary as we need to collect output from each process.
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    count = 1
    for file_list in list_of_file_list:
        p = multiprocessing.Process(target=compute_liwc_categories_from_filelist,
                                    args=(file_list, src_dir_path, user_name_list,return_dict,complete_network,in_network,only_week),
                                    name=count)
        process_list.append(p)
        p.start()
        count = count + 1

    for process in process_list:
        process.join()

    for process, category_dicts_list in return_dict.items():
        total_sent_category_dict.update(category_dicts_list[0])
        total_recv_category_dict.update(category_dicts_list[1])
        total_within_category_dict.update(category_dicts_list[2])


    return total_sent_category_dict, total_recv_category_dict, total_within_category_dict




if __name__ == "__main__":
    # ===========================================================================================================
    # ===========Computing LIWC categories multiprocess=============================
    # ===========================================================================================================

    book_list = ["MENG"]
    employee_list = employee.employees_given_book_list(book_list)
    total_sent_category_dict, total_recv_category_dict, total_within_category_dict = \
        compute_liwc_categories_from_filelist_multiproc(cfg.SENTIMENT_BUSINESS,employee_list,1,complete_network=False,in_network=True,
                                                        start_week=127, end_week=127,only_week=True)


    print(total_sent_category_dict)
    print(total_recv_category_dict)
    print(total_within_category_dict)

    pass
