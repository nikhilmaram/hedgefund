import string

import config as cfg

CAT_DELIM = "%"

# =========================================================================
# ==================== LIWC Categorization=================================
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



if __name__ == "__main__":
    gettysburg = '''Four score and seven years ago our fathers brought forth on
      this continent a new nation, conceived in liberty, and dedicated to the
      proposition that all men are created equal. Now we are engaged in a great
      civil war, testing whether that nation, or any nation so conceived and so
      dedicated, can long endure. We are met on a great battlefield of that war.
      We have come to dedicate a portion of that field, as a final resting place
      for those who here gave their lives that that nation might live. It is
      altogether fitting and proper that we should do this.'''

    liwc = parse_liwc(cfg.LIWC_DICTIONARY)
    counter_dict = extract(liwc,gettysburg,percentage=True)
    print(counter_dict)

