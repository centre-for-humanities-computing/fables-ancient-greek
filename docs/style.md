# Style

The corpus can be investigated from a number of stylistical perspectives.

## Vocabulary Richness

The vocabulary richness of each fable was calculated on the lemmatized work both with vanilla type-token ratio, but also with moving windows of size 10 and 50.

<figure>
<iframe src="../_static/vocabulary_richness.html" width="2450px" height="1050px"></iframe>
<figcaption>Use of Terms on a Group and Individual Level</figcaption>
</figure>

## UPOS Tags

UPOS tags were tallied up in all fables without removal of any stop words or lemmatization.

<figure>
<iframe src="../_static/upos_scatter_matrix.html" width="2450px" height="1050px"></iframe>
<figcaption>Relative Frequencies of Nouns, Verbs and Adjectives in all Fables</figcaption>
</figure>

<figure>
<iframe src="../_static/upos_scatter_matrix_3d.html" width="2450px" height="1050px"></iframe>
<figcaption>Relative Frequencies of Nouns, Verbs and Adjectives in all Fables</figcaption>
</figure>

<figure>
<iframe src="../_static/upos_scatter_matrix_function.html" width="2450px" height="1050px"></iframe>
<figcaption>Relative Frequencies of Function Word Categories</figcaption>
</figure>

<figure>
<iframe src="../_static/upos_wave_plot.html" width="2450px" height="1050px"></iframe>
<figcaption>Wave Plot of UPOS Tag Distributions</figcaption>
</figure>

The most frequent 2 to 4-grams of UPOS tags were also counted for each work.

<figure>
<iframe src="../_static/upos_patterns.html" width="2450px" height="1400px"></iframe>
<figcaption>Most Frequent N-grams of UPOS tags</figcaption>
</figure>

The most frequent 4-grams of UPOS tags were also counted for each work.

<figure>
<iframe src="../_static/upos_patterns_4.html" width="2450px" height="1400px"></iframe>
<figcaption>Most Frequent N-grams of UPOS tags</figcaption>
</figure>

## Lengths

The length of fables (number of tokens), average length of tokens and mean sentence length were calculated for each work. Texts were split on punctuation to create the sentences. 
Punctuation was defined as full stops (.) and Greek question marks (;). Commas and elevated dots were not counted as punctuation. 
Metrical feet, metre, and stanzas were not taken into account.  

<figure>
<iframe src="../_static/length_scatter_matrix.html" width="2450px" height="1050px"></iframe>
<figcaption>Lengths in all Fables per Group</figcaption>
</figure>

<figure>
<iframe src="../_static/lengths_3d.html" width="2450px" height="1050px"></iframe>
<figcaption>Total number of words (length), number of unique words (n_types), and number of unique lemmata (n_lemmata) in all Fables per Group</figcaption>
</figure>


## Vocabulary Richness (Noun, Adj, Verb)

The vocabulary richness of each fable was calculated on the lemmatized work both with vanilla type-token ratio, but also with moving windows of size 10 and 50.

<figure>
<iframe src="../_static/vocabulary_richness_noun_adj_verb.html" width="2450px" height="1050px"></iframe>
<figcaption>Use of Terms on a Group and Individual Level</figcaption>
</figure>
