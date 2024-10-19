import time 
import random 


class Corpus:
    """
    Reads a file (where each line represents a document and words are separated by white space).
    """

    # Initialize the Corpus class with a list of documents.
    def __init__(self, docs):
        self.docs = docs

    # Class method to create a Corpus object by reading a file.
    @classmethod
    def read(cls, filename, dtype=str, num_samples=None):
        start_time = time.time()
        docs = []  # Initialize an empty list to store documents.
        
        # Open the specified file in read mode.
        with open(filename, 'r') as f:
            for line in f:
                # Split each line into words using whitespace as the delimiter.
                doc = [dtype(w.strip()) for w in line.split( )]
                
                # If the document has more than zero words, add it to the list of documents.
                if len(doc) > 0:
                    docs.append(doc)
                # print(docs)  # Debugging: Print the list of documents at each iteration.
         # If num_samples is specified and less than the length of docs, sample the docs
        if num_samples is not None and num_samples < len(docs):
            docs = random.sample(docs, num_samples)
        # Create a Corpus object with the list of documents and return it.
        
        end_time = time.time()
        print(f"Time taken for reading file in corpus class is {filename}: {end_time - start_time} seconds")
        return Corpus(docs)

    # Iterator method to iterate over documents in the Corpus.
    def __iter__(self):
        for doc in self.docs:
            yield doc

    # Get the number of documents in the Corpus.
    def __len__(self):
        return len(self.docs)
