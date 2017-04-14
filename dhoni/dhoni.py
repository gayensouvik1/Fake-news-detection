# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import stopwords # Import the stop word list
print stopwords.words("english")


# import re

# example1 = "Absolutely magnificent, DHONI! Finishes off in style, a magnificent strike into the crowd, India lift the World Cup after 28 years, the party start in the dressing room and its an Indian captain who’s been absolutely magnificent in the night of the final."
# # Use regular expressions to do a find-and-replace
# letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
#                       " ",                   # The pattern to replace it with
#                       example1 )  # The text to search
# print letters_only