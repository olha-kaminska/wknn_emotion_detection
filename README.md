## Nearest neighbour approaches for Emotion Detection in Tweets
Code for the paper written by [Olha Kaminska](https://scholar.google.com/citations?hl=en&user=yRgJkEwAAAAJ), [Chris Cornelis](https://scholar.google.com/citations?hl=en&user=ln46HlkAAAAJ), and [Veronique Hoste](https://scholar.google.com/citations?hl=en&user=WxOsW3IAAAAJ) presented at [EACL 2021](https://2021.eacl.org/) during [WASSA workshop](https://wt-public.emm4u.eu/wassa2021/) as [a poster](https://olha-kaminska.github.io/WASSA2021_poster_Olha_Kaminska.pdf).

This paper uses data from <a href="https://competitions.codalab.org/competitions/17751">SemEval-2018 Task 1: Affect in Tweets</a> competition, where we participated in Task 2a "EI-oc" (emotion intensity ordered classification) for English tweets. 

### Repository Overview ###
- The **code** directory contains .py files with different functions:
  - *preprocessing.py* - functions for data uploading and preperation;
  - *embeddings_and_lexicons.py* - functions for tweets embeddings with different methods and lexicons;
  - *wknn_eval.py* - functions for wkNN approach and cross-validation evaluation.
- The **data** directory contains *README_data_download.md* file with instruction on uploading necessary dataset files that should be saved in the *data* folder.
- The **lexica** directory contains *README_lexicons_download.md* file with instruction on uploading necessary lexicons files that should be saved in the *lexica* folder.
- The **model** directory contains *README_model_download.md* file with instruction on uploading necessary models that should be saved in the *model* folder.
- The file **Example.ipynb** provides an overview of all function and their usage on the example of Anger dataset. It is built as a pipeline described in the paper with corresponded results. 
- The file *requirements.txt* contains the list of all necessary packages and versions used with the Python 3.7.4 environment.
- The file *WASSA2021_poster_Olha_Kaminska.pdf* contains a poster that was presented for this paper at WASSA 2021.

### ACL link ###
https://www.aclweb.org/anthology/2021.wassa-1.22/  

### Arxiv link ###
https://arxiv.org/abs/2107.05394  

### Abstract ###
*Emotion detection is an important task that can be applied to social media data to discover new knowledge. While the use of deep learning methods for this task has been prevalent, they are black-box models, making their decisions hard to interpret for a human operator. Therefore, in this paper, we propose an approach using weighted k Nearest Neighbours (kNN), a simple, easy to implement, and explainable machine learning model. These qualities can help to enhance results' reliability and guide error analysis. In particular, we apply the weighted kNN model to the shared emotion detection task in tweets from SemEval-2018. Tweets are represented using different text embedding methods and emotion lexicon vocabulary scores, and classification is done by an ensemble of weighted kNN models. Our best approaches obtain results competitive with state-of-the-art solutions and open up a promising alternative path to neural network methods.*

### BibTeX citation: ###
>@inproceedings{kaminska2021nearest,
  title={Nearest neighbour approaches for Emotion Detection in Tweets},
  author={Kaminska, Olha and Cornelis, Chris and Hoste, Veronique},
  booktitle={Proceedings of the Eleventh Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
  pages={203--212},
  year={2021}
  }
