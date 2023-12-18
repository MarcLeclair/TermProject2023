# README for Enhanced Question Generation Project

## Overview
This project builds upon the foundational work in question generation by expanding the dataset scope and introducing machine-generated summaries. Our goal is to enhance the performance and applicability of question generation models in educational contexts.

## Problem Statement
The original study, "A Feasibility Study of Answer-Agnostic Question Generation for Education," utilized a limited set of textbook chapters and their summaries. Our project extends this approach by incorporating the BookSum dataset, which offers a more diverse range of texts, to increase the accuracy and generalizability of the question generation model.

## Dataset
We employed the [BookSum dataset](https://huggingface.co/datasets/kmfoda/booksum), which includes a wide array of book summaries. This rich dataset enabled us to generate machine summaries, thereby providing a broader base for our question generation model.

## Usage
To use this project, users will need to run the `get_booksum.py` script. This script is designed to process the BookSum dataset and generate machine-made summaries that are then used for question generation. You will also need to download the book files from [here](https://storage.cloud.google.com/sfr-books-dataset-chapters-research/all_chapterized_books.zip). Those book files then can be linked to the dataset used in `get_booksum.py`.

## Original Repository
The project is based on the original repository by Liam Dugan, which can be accessed at [https://github.com/liamdugan/summary-qg](https://github.com/liamdugan/summary-qg). Thanks to the author for laying the groundwork for this research.

## Acknowledgements
We would like to acknowledge the authors of the [BookSum dataset](https://github.com/salesforce/booksum) and the author of the dataset on hugging face.As well as Liam Dugan for their significant contributions to the field of natural language processing and educational technology.
