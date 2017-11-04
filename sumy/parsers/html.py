# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from breadability.readable import Article
from ..utils import cached_property, fetch_url
from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from .parser import DocumentParser


class HtmlParser(DocumentParser):
    """Parser of text from HTML format into DOM."""
    SIGNIFICANT_TAGS = (
        "h1", "h2", "h3",
        "b", "strong",
        "big",
        "dfn",
        "em",
    )

    @classmethod
    def from_string(cls, string, url, tokenizer):
        return cls(string, tokenizer, url)

    @classmethod
    def from_file(cls, file_path, url, tokenizer):
        with open(file_path, "rb") as file:
            return cls(file.read(), tokenizer, url)

    @classmethod
    def from_url(cls, url, tokenizer):
        data = fetch_url(url)
        return cls(data, tokenizer, url)

    def __init__(self, html_content, tokenizer, url=None):
        super(HtmlParser, self).__init__(tokenizer)
        self._article = Article(html_content, url)

        #count number of paragraphs on page
        self.desired_ct = 2
        self.paragraph_ct = 0
        self.correct_paragraph_ct = 0

        #count paragraphs
        for paragraph in self._article.main_text:
            #check if paragraph is longer than 6 words
            if len(self._article.main_text[self.paragraph_ct]) >=6:
                self.paragraph_ct+=1
            else:
                #trim short paragraphs that will skew data
                del self._article.main_text[self.paragraph_ct]

        #check if a lot of paragraphs, usually intro material will be in the first couple paragraphs
        if self.paragraph_ct > self.desired_ct:
            #calculate number of paragraphs to use
            self.correct_paragraph_ct = int(self.paragraph_ct * 0.5 // 10)
            #if correction results in too few paragraphs - opt to use desired number
            if self.correct_paragraph_ct <= 1:
                self.correct_paragraph_ct = self.desired_ct
        #if less than desired and more than 0, already at correct paragraph count
        elif self.paragraph_ct >0:
            self.correct_paragraph_ct = self.paragraph_ct

        #delete excess paragraphs
        diff = self.paragraph_ct - self.correct_paragraph_ct
        while diff > 0:
            del self._article.main_text[diff+self.correct_paragraph_ct-1]
            diff-=1

        print(self._article.main_text)

    @cached_property
    def significant_words(self):
        words = []
        for paragraph in self._article.main_text:
            for text, annotations in paragraph:
                if self._contains_any(annotations, *self.SIGNIFICANT_TAGS):
                    words.extend(self.tokenize_words(text))

        if words:
            return tuple(words)
        else:
            return self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self):
        words = []
        for paragraph in self._article.main_text:
            for text, annotations in paragraph:
                if self._contains_any(annotations, "a", "strike", "s"):
                    words.extend(self.tokenize_words(text))

        if words:
            return tuple(words)
        else:
            return self.STIGMA_WORDS

    def _contains_any(self, sequence, *args):
        if sequence is None:
            return False

        for item in args:
            if item in sequence:
                return True

        return False

    @cached_property
    def document(self):
        # "a", "abbr", "acronym", "b", "big", "blink", "blockquote", "cite", "code",
        # "dd", "del", "dfn", "dir", "dl", "dt", "em", "h", "h1", "h2", "h3", "h4",
        # "h5", "h6", "i", "ins", "kbd", "li", "marquee", "menu", "ol", "pre", "q",
        # "s", "samp", "strike", "strong", "sub", "sup", "tt", "u", "ul", "var",

        annotated_text = self._article.main_text

        paragraphs = []
        for paragraph in annotated_text:
            sentences = []

            current_text = ""
            for text, annotations in paragraph:
                if annotations and ("h1" in annotations or "h2" in annotations or "h3" in annotations):
                    sentences.append(Sentence(text, self._tokenizer, is_heading=True))
                # skip <pre> nodes
                elif not (annotations and "pre" in annotations):
                    current_text += " " + text

            new_sentences = self.tokenize_sentences(current_text)
            sentences.extend(Sentence(s, self._tokenizer) for s in new_sentences)
            paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)
