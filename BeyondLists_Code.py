#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
pd.set_option('display.max_rows', 100)
import xml.etree.ElementTree as ET
from collections import Counter
from collections import defaultdict
import plotly.express as px
import numpy as np


# # 🟧 Corpus preparation
# The corpus preparation consists of creating three lists ('bags-of-words'): the first one (called 'target_words') contains all the words of the target corpus, the second one ('reference_words') contains all the words of the reference corpus and the third one ('target_books') contains all the words of the target corpus divided per book. For this study, I used lemma of each word (to reduce morphological variation) associated with its part-of-speech tag.
# 
# The link to the file to process is (last access 29/3/26): https://github.com/lascivaroma/latin-lemmatized-texts/blob/main/lemmatized/xml/urn%3Acts%3AlatinLit%3Aphi0978.phi001.perseus-lat2.xml

# In[2]:


tree = ET.parse("/Users ...... urn-cts-latinLit-phi0978.phi001.perseus-lat2.xml")


# In[3]:


target_words = [] ## create a bag-of-words for the target corpus
reference_words = [] ## create a bag-of-words for the reference corpus
target_books = defaultdict(list) ## creare a bag-of-words for each geo book

root = tree.getroot()
ns = {"tei": "http://www.tei-c.org/ns/1.0"}

for chapter in root.findall(".//tei:ab[@type='chapter']", ns): ## for each chapter in the .xml

    cts = chapter.get("n")
    book_number = int(cts.split(":")[-1].split(".")[0])  # get the book number

    if book_number in [3, 4, 5, 6]: ## if the book is one of the target books

        for w in chapter.findall(".//tei:w", ns):

            if w.get("pos") != "PUNC": ## exclude punctuation
                lemma = w.get("lemma") ## get the lemma
                pos = w.get("pos") ## get the pos
                lemma_pos = (lemma+"_"+pos) ## avoid merging lemma with different pos (?)

                target_words.append(lemma_pos) ## put it in the target bag-of-words
                target_books[book_number].append(lemma_pos) ## put it also in the corresponding geo-book bag-of-words

    else: ## if the book is not one of the target books

        for w in chapter.findall(".//tei:w", ns):

            if w.get("pos") != "PUNC":
                lemma = w.get("lemma")
                pos = w.get("pos")
                lemma_pos = (lemma+"_"+pos)

                reference_words.append(lemma_pos) ## put in the reference bag-of-words


# In[4]:


target_books[4] ## print the bag-of-words of book 4


# # 🟫 Compute raw frequencies
# The raw frequency of each lemma_pos in the target and reference corpus is computed.

# In[5]:


## count tot frequencies for each word in the targ and ref corpus
target_freq = Counter(target_words)
reference_freq = Counter(reference_words)


# In[6]:


target_freq ## list of unique lemmas in the target corpus with frequency


# In[7]:


book_freq = {book: Counter(words) for book, words in target_books.items()} ## list of unique lemmas in each book with frequency


# In[8]:


book_freq[4]


# In[9]:


book_sizes = {b: len(words) for b,words in target_books.items()} ## total size of each book
book_sizes[4]


# # 🟥 Preliminary analysis
# - compute the total size of the target and reference corpus
# - compute unique lemmas in the target corpus
# - plot the target corpus's unique lemmas and tokens distribution per POS

# In total, the target corpus contains 39,923 tokens excluding punctuation.

# In[10]:


total_target_words = len(target_words)
total_target_words


# In total, the reference corpus contains 364,571 tokens excluding punctuation.

# In[11]:


total_reference_words = len(reference_words)
total_reference_words


# In total, the target corpus contains 10,210 unique lemmas.

# In[12]:


len(target_freq)


# The pie charts below shows the distribution of unique lemmas and tokens by POS (for a discussion of the results see the related paper).

# In[13]:


## Compute unique lemma distribution per POS

pos_uniquelemma = defaultdict(int)
for lemma_pos in target_freq: ## for each unique lemma
    lemma, pos = lemma_pos.rsplit('_', 1)
    pos_uniquelemma[pos] += 1 ## add +1 at the corresponding POS group

sorted_pos = sorted(pos_uniquelemma.keys())
labels = sorted_pos
sizes = [pos_uniquelemma[pos] for pos in sorted_pos]

color_map = {
    'NOMpro': '#4E79A7',
    'NOMcom': '#F28E2B',
    'VER': '#E15759',
    'ADJqual': '#76B7B2',
    'ADV': '#59A14F',
    'PRE': '#EDC948',
    'CON': '#B07AA1',
    'PROdem': '#FF9DA7',
    'PROrel': '#9C755F',
}

fig1 = px.pie(
    names=labels,
    values=sizes,
    title="Unique lemmas distribution by POS",
    width=600,
    height=500,
    color=labels,
    color_discrete_map=color_map
)
fig1.update_traces(
    textinfo='none',
    hovertemplate='<b>%{label}</b><br>Unique lemmas: %{value}<br>Percentage: %{percent}'
)

fig1.show()


# In[14]:


## Compute tokens distribution per POS

pos_tokens = Counter()
for lemma_pos, freq in target_freq.items(): ## for each unique lemma
    pos = lemma_pos.rsplit('_', 1)[-1]
    pos_tokens[pos] += freq ## add the lemma frequency to the POS group

sizes_freq = [pos_tokens.get(pos, 0) for pos in sorted_pos]

fig2 = px.pie(
    names=labels,
    values=sizes_freq,
    title="Token distribution by POS",
    width=600,
    height=500,
    color=labels,
    color_discrete_map=color_map
)
fig2.update_traces(
    textinfo='none',
    hovertemplate='<b>%{label}</b><br>Token count: %{value}<br>Percentage: %{percent}'
)

fig2.show()


# # 🟪 Keyword analysis

# ## 1. Fisher's test

# Python’s scipy library was employed for all statistical computations. For the application of the Fisher’s test scipy library in linguistic analysis see Porter 2021, Huang 2023. The alternative parameter was set to 'greater' specifying a one-sided test. This configuration assesses whether the frequency of a given word in the target corpus is significantly higher than would be expected under the null hypothesis. The use of a directional test is particularly suitable for keyword analysis, as it enables the identification of overrepresented lexical items.

# In[15]:


from scipy.stats import fisher_exact
fisher_scores = {}
fisher_p_scores = {}
min_count = 1 ## the min count is set to 1, sometimes in KY analysis a higher min count is used

for word in target_freq: ## for each unique lemma
    if target_freq[word] < min_count:
        continue

    fisher = fisher_exact([[target_freq[word], (total_target_words - target_freq[word])], [reference_freq[word], total_reference_words - reference_freq[word]]], alternative='greater')
    fisher_scores[word] = fisher.statistic
    p_value_from_fisher = fisher.pvalue

    if p_value_from_fisher < 0.05:
        fisher_p_scores[word] = '< 0.05'
    else: fisher_p_scores[word] = '> 0.05'

sorted_fisher_p = dict(sorted(fisher_p_scores.items(), key=lambda x: x[1], reverse=False))
list(sorted_fisher_p.items())[:10]


# ## 2. Difference of proportions (ΔP)

# In case a word is not attested in the ref corpus, the value of 'reference_freq[word]' is automatically set to 0. The deltapi_score thus is equal to the p_target. However, since several words occurr only once in the target corpus, their deltapi is still relatively low.

# In[16]:


deltapi_scores = {}
min_count = 1 ## the min count is set to 1, sometimes in KY analysis a higher min count is used

for word in target_freq: ## for each unique lemma
    if target_freq[word] < min_count:
        continue

    p_target = target_freq[word] / total_target_words

    ## calculate ΔP
    deltapi = (p_target) - (reference_freq[word] / total_reference_words)
    deltapi_scores[word] = deltapi

## sort by ΔP score in descending order
highest_deltapi = dict(sorted(deltapi_scores.items(), key=lambda x: x[1], reverse=True))
list(highest_deltapi.items())[:10]


# The table below contains the list of all the unique lemmas in the target corpus ranked by deltapi and with the p_score.

# In[17]:


sorted_deltapi = list(highest_deltapi.items())
df = pd.DataFrame(sorted_deltapi, columns=["lemma_pos", "deltapi"])
df["absolute_freq_tar"] = df["lemma_pos"].apply(lambda word: (target_freq[word]))
for b in [3, 4, 5, 6]:
    df[f"freq_book{b}"] = df["lemma_pos"].apply(
        lambda word: book_freq[b].get(word, 0)
    )
df["absolute_freq_ref"] = df["lemma_pos"].apply(lambda word: (reference_freq[word]))
df["relative_freq_tar"] = df["lemma_pos"].apply(lambda word: (target_freq[word] / total_target_words) * 100)
df["relative_freq_ref"] = df["lemma_pos"].apply(lambda word: (reference_freq[word] / total_reference_words) * 100)
df["p_value_fisher"] = df["lemma_pos"].apply(lambda word: fisher_p_scores[word])
df[["lemma", "pos"]] = df["lemma_pos"].str.rsplit("_", n=1, expand=True)
df = df.drop(columns=["lemma_pos"])
df = df.reindex(columns=["lemma", "pos", "absolute_freq_tar", "freq_book3", "freq_book4", "freq_book5", "freq_book6", "absolute_freq_ref", "deltapi", "relative_freq_tar", "relative_freq_ref", "p_value_fisher"])
df.head(10)


# # 🟨 Analysis

# ### Preliminary analysis

# In total, 1,671 lemmas have significant p < 0.05. Most of the keywords are NOMpro (931) followed by NOMcom (226).

# In[18]:


significant_df = df[df["p_value_fisher"] == "< 0.05"]
len(significant_df)


# In[19]:


significant_df["pos"].value_counts().head(10)


# ### 1. Analyse the most frequent keywords in the target corpus

# In[21]:


significant_df = significant_df.sort_values(
    by="absolute_freq_tar",
    ascending=False
)

top_significant = significant_df.head(15)

fig = px.bar(
    top_significant,
    x="lemma",
    y="absolute_freq_tar",
    color="pos",
    color_discrete_map=color_map,
    title="15 most frequent keywords",
    width=650,
    height=600,
)

fig.update_layout(
    template="plotly_white",
    xaxis_tickangle=-60,
    xaxis=dict(categoryorder='total descending'),
    margin=dict(t=80, b=150, l=60, r=60),
    title_font_size=16,
    legend_title_text="POS",
    showlegend=True
)
fig.update_yaxes(title="absolute frequency targ")


# Inspect the keywords by POS.

# In[23]:


significant_df[significant_df["pos"] == "NOMcom"].head(10)


# ### 2. Analyse the distribution of keywords and not-keywords in the target corpus

# The table below shows the ratio between the number of keyword lemmas and the total lemmas per POS in the target corpus.

# In[24]:


pos_keywords = significant_df["pos"].value_counts().to_frame("keyword_lemmas")
pos_keywords["total_lemmas"] = pos_keywords.index.map(pos_uniquelemma)
pos_keywords["ratio"] = (
    pos_keywords["keyword_lemmas"] / pos_keywords["total_lemmas"] * 100
)
pos_keywords = pos_keywords.sort_values("ratio", ascending=False)
pos_keywords.head(10)


# In total, 8,539 lemmas were marked as not significant.

# In[25]:


notsignificant_df = df[df["p_value_fisher"] == "> 0.05"]
len(notsignificant_df)


# Inspect non-keywords by POS.

# In[27]:


notsignificant_df = notsignificant_df.sort_values(
    by="absolute_freq_ref",
    ascending=False ## sort by the abs freq in the ref corpus
)
notsignificant_df[notsignificant_df["pos"] == "NOMpro"]


# The plot below shows the distribution of keywords (in red) and not-keywords (in blue) by absolute frequency in the target corpus (x-axis) and delta-pi (y-axis).

# In[30]:


df["keyword"] = df["p_value_fisher"] == "< 0.05"
df = df.sort_values("keyword", ascending=True)

df["color_group"] = df["keyword"].map({
    True: "keyword",
    False: "not keyword"
})
fig = px.scatter(
    df,
    x="absolute_freq_tar",
    y="deltapi", 
    color="color_group",
    color_discrete_map={
        "keyword": "#d62728",
        "not keyword": "#4E79A7"
    },
    hover_name="lemma",
    hover_data={
        "pos": True,
        "absolute_freq_tar": True,
        "absolute_freq_ref": True,
        "deltapi": False
    }
)
fig.update_layout(
    template="simple_white",
    xaxis_title="absolute frequency targ (log)",
    yaxis_title="delta-pi",
    width=800,
    height=600,
    legend_title=""
)
fig.update_xaxes(
    type="log",
    tickvals=[1, 10, 100, 1000],
    ticktext=["1", "10", "100", "1000"]
)
fig.add_hline(
    y=0,
    line_dash="dash",
    line_width=1,
    line_color="grey"
)
fig.update_traces(marker=dict(opacity=0.8, size=8, line=dict(width=0.2, color='DarkSlateGrey')))
fig.show()


# # 🟦 In-context analysis

# In[90]:


sub_targ_verbs = []

root = tree.getroot()
ns = {"tei": "http://www.tei-c.org/ns/1.0"}

stop = False

for chapter in root.findall(".//tei:ab[@type='chapter']", ns):

    if stop:
        break

    cts = chapter.get("n")
    book_number = int(cts.split(":")[-1].split(".")[0])

    if book_number == 4:
        
        for w in chapter.findall(".//tei:w", ns):
            lemma = w.get("lemma")
            pos = w.get("pos")

            if pos == "VER":
                lemma_pos = f"{lemma}_{pos}"
                sub_targ_verbs.append(lemma_pos)
                
            if lemma == "peleo": ## manual restriction to the selected subtarget corpus
                stop = True
                break 

    else: continue


# In total, the subtarget corpus contains 222 unique verbs.

# In[93]:


subtarget_freq = Counter(sub_targ_verbs)
len(subtarget_freq)


# In[94]:


df_subtarget = pd.DataFrame(subtarget_freq.items(), columns=['verb', 'frequency'])
df_subtarget = df_subtarget.sort_values(by='frequency', ascending=False).reset_index(drop=True)


# In[95]:


def significance_label(word):
    if word in fisher_p_scores:
        if fisher_p_scores[word] == "< 0.05":
            return "keyword"
        else:
            return "not keyword"
    else:
        return None

df_subtarget['significance'] = df_subtarget['verb'].apply(significance_label)
df_subtarget["verb"] = df_subtarget["verb"].str.split("_").str[0]
df_subtarget.head(10)


# In[98]:


significant_df_subtarget = df_subtarget[df_subtarget["significance"] == "keyword"]
len(significant_df_subtarget)


# In[110]:


df_subtarget[df_subtarget["verb"] == "labor1"]


# In[100]:


top_significant = df_subtarget.head(15)

fig = px.bar(
    top_significant,
    x="verb",
    y="frequency",
    color="significance",
    title="15 most frequent verbs",
    width=650,
    height=600,
    color_discrete_map = {
    "keyword": "#d62728",   
    "not keyword": "#4E79A7"  
    }
)

fig.update_layout(
    template="plotly_white",
    xaxis_tickangle=-60,
    yaxis_title="absolute frequency",
    xaxis=dict(categoryorder="total descending"),
    margin=dict(t=80, b=150, l=60, r=60),
    title_font_size=16,
    legend_title_text="",
    showlegend=True
)


# In[103]:


significant_df_subtarget

