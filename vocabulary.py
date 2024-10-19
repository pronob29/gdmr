import time 

class Vocabulary:
    """
    Manage vocabulary
    """

    def __init__(self):
        # Initialize three instance variables to manage the vocabulary.
        self.vocas = []    # id to word
        self.vocas_id = {} # word to id
        self.docfreq = []  # id to document frequency

    def read_corpus(self, corpus):
        start_time = time.time()
        result = []
        # Iterate through each document in the corpus.
        for doc in corpus:
            # Convert each document into a list of word IDs using doc_to_ids method.
            result.append(self.doc_to_ids(doc))
        # Return a list of lists, where each inner list represents a document with word IDs.
        end_time = time.time()
        print(f"Time taken for reading file in vocabulary class is {end_time - start_time} seconds")
        return result

    def term_to_id(self, term):
        # Convert a term (word) to a word ID.
        if term not in self.vocas_id:
            # If the term is not in the vocabulary, assign a new word ID to it.
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            # Add the term to the vocabulary.
            self.vocas.append(term)
            # Initialize its document frequency to 0.
            self.docfreq.append(0)
        else:
            # If the term is already in the vocabulary, retrieve its existing word ID.
            voca_id = self.vocas_id[term]
        # Return the word ID.
        return voca_id

    def doc_to_ids(self, doc):
        result = []
        words = dict()
        # Iterate through each term in the document.
        for term in doc:
            # Convert the term to a word ID using term_to_id method.
            tid = self.term_to_id(term)
            # Append the word ID to the result list.
            result.append(tid)
            if not tid in words:
                words[tid] = 1
                # Update the document frequency of the term.
                self.docfreq[tid] += 1
        # Return a list of word IDs representing the document.
        return result

    def cut_low_freq(self, corpus, threshold = 4):
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        # Iterate through the existing vocabulary.
        for tid, term in enumerate(self.vocas):
            freq = self.docfreq[tid]
            # Check if the document frequency of the term is above the threshold.
            if freq > threshold:
                # If above threshold, assign a new word ID to the term in the new vocabulary.
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                # Add the term to the new vocabulary.
                new_vocas.append(term)
                # Update the document frequency in the new document frequency list.
                new_docfreq.append(freq)
                # Create a mapping from old word IDs to new word IDs.
                conv_map[tid] = new_id
        # Update the vocabulary and document frequency lists with the new ones.
        self.vocas = new_vocas
        self.docfreq = new_docfreq

        def conv(doc):
            new_doc = []
            # Convert document to use the new word IDs based on the conv_map.
            for id in doc:
                if id in conv_map:
                    new_doc.append(conv_map[id])
            return new_doc

        # Convert documents in the corpus to use the new word IDs.
        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        # Retrieve a word from the vocabulary using its ID.
        return self.vocas[v]

    def size(self):
        # Get the size (number of words) of the vocabulary.
        return len(self.vocas)
