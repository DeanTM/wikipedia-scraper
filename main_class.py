import wikipedia
import time
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from enum import Enum


class SelectBy(Enum):
    RANDOM = 1
    GREEDY = 2
    RANDOMLOCAL = 3


# some functions which may need to be replaced if we switch out
# the wikipedia api
def get_page_title(page):
    return page.title

def get_page_summary(page):
    return page.summary

def get_page_content(page):
    return page.content

def get_page_neighbours(page):
    return page.links

# this class is the main body of the project
class ratings_state():
    @classmethod
    def load(pickle_folder, selection=SelectBy.GREEDY):
        # check path is valid
        from os.path import exists
        if not exists(pickle_folder):
            raise Exception('Folder %s does not exist.' % pickle_folder)
        import pickle        
        rs = ratings_state(selection=selection)
        # update default save address
        rs.save_address_state = pickle_folder + '/state.pickle'
        rs.save_address_words = pickle_folder + '/words.pickle'
        # load properties
        f = open(rs.save_address_state, 'rb')
        rs.ratings = pickle.load(f)
        f.close()
        f = open(rs.save_address_words, 'rb')
        rs.words = pickle.load(f)
        f.close()
        rs.rated_count = len([x for x, y in rs.ratings.items() if y['rated'] != 0])
        return rs

    def __init__(self, initial='Machine learning', selection=SelectBy.GREEDY):
        import os
        if not os.path.exists('wiki_state_files'):
            os.makedirs('wiki_state_files')
        self.ratings = {}  # {title: {'features':(idx, count), 'rated':0}}
        self.words = {}  # {word: (index, count, doc_count)}
        self.word_index = {}  # {idx: word}
        self.alpha = 0.1  # bias for doc count
        self.save_address_state = 'wiki_state_files/state.pickle'
        self.save_address_words = 'wiki_state_files/words.pickle'
        self.initial = initial
        self.lemmatizer = WordNetLemmatizer()
        self.rated_count = 0
        self.selection = selection

    def run(self):
        if len(self.ratings) == 0:
            page = self.open_article(self.initial)
            features = self.update_feature_extraction(page)
            self.ratings[self.initial] = {'features': features, 'rated': 0}
            self.present_summary(page)
            response = self.get_response(page)
            if response == 'full':
                self.present_full_article(page)
                response = self.get_full_response(page)
            self.update_with_response(response, page)
        while True:
            page = self.recommend_title()
            self.present_summary(page)
            response = self.get_response(page)
            if response == 'full':
                self.present_full_article(page)
                response = self.get_full_response(page)
            self.update_with_response(response, page)

    def update_with_response(self, response, page):
        if response == 'dislike':
            self.ratings[get_page_title(page)]['rated'] = -1
        elif response == 'full_dislike':
            self.ratings[get_page_title(page)]['rated'] = -2
        elif response == 'like':
            self.ratings[get_page_title(page)]['rated'] = 1
            self.scrape_and_add_neighbours(
                    get_page_title(page)
                    )
        elif response == 'full_like':
            self.ratings[get_page_title(page)]['rated'] = 2
            self.scrape_and_add_neighbours(
                    get_page_title(page)
                    )

    def get_response(self, page):
        query_text = "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"\
            + "\ntype 'save' to save"\
            + "\ntype 'full' to read full article"\
            + "\ntype 'like' to like article"\
            + "\ntype 'dislike' to dislike article"\
            + "\ntype 'reload' to see the summary again"\
            + "\n>>>"
            # + "\ntype 'random' to select articles randomly"
        response = input(query_text)
        if response.lower() not in [
            'save', 'full', 'like', 'dislike', 'random', 'reload']:
            response = input('Invalid response...\n' + query_text)
        if response.lower() == 'reload':
            self.present_summary(page)
            return self.get_response(page)
        elif response.lower() == 'save':
            self.save_state()
            response = input('Saving complete...\n' + query_text)
        return response  # like, dislike, full

    def get_full_response(self, page):
        query_text = "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"\
            + "\ntype 'save' to save"\
            + "\ntype 'like' to like article"\
            + "\ntype 'full_like' to really like the article"\
            + "\ntype 'dislike' to dislike article"\
            + "\ntype 'full_dislike' to really like the article"\
            + "\ntype 'reload' to see the page again"\
            + "\n>>>"
            # + "\ntype 'random' to select articles randomly"
        response = input(query_text)
        if response.lower() not in [
            'save', 'full', 'like', 'dislike',
            'random', 'reload', 'full_like', 'full_dislike']:
            response = input('Invalid response...\n' + query_text)
        if response.lower() == 'reload':
            self.present_full_article(page)
            return self.get_full_response(page)
        elif response.lower() == 'save':
            self.save_state()
            response = input('Saving complete...\n' + query_text)
        return response  # like, dislike, full_like, full_dislike

    def save_state(self):
        import pickle
        f = open(self.save_address_state, 'wb')
        pickle.dump(self.ratings, f)
        f.close()
        f = open(self.save_address_words, 'wb')
        pickle.dump(self.words, f)
        f.close()
    
    def open_article(self, article_title):
        try:
            page = wikipedia.WikipediaPage(article_title)
        except:
            raise Exception('Failed to load page.')
        return page

    def _recommend_by_best_rating(self):
        best_rating = (None, -3)
        for title, info in self.ratings.items():
            # only predict on unseen articles
            if info['rated'] == 0:
                predicted_rating = self.predict_rating(info['features'])
                if predicted_rating > best_rating[1]:
                    best_rating = (title, predicted_rating)
        if best_rating[0] is None:
            # there are no unseen crawled articles
            # we must crawl from new articles
            # we'll crawl from the best negative articles
            print('No unrated articles found.\nSearching for new articles.')
            potential_starts = list(
                (k, self.predict_rating(v['features']))
                for k, v in self.ratings.items()
                )
            potential_starts = sorted(potential_starts, key=lambda x: x[1])
            has_found_new = False
            while potential_starts and not has_found_new:
                next_candidate = potential_starts.pop()
                has_found_new = self.scrape_and_add_neighbours(
                    next_candidate[0]
                    )
            if not has_found_new:
                raise Exception('Could not find any new articles')
            # predict again on the newly added articles
            return self._recommend_by_best_rating()
        else:
            return self.open_article(best_rating[0])

    def recommend_title(self):
        if len(self.ratings) == 0:
            return self.open_article(self.initial)
        if self.selection == SelectBy.GREEDY:
            # recommend by best rating
            return self._recommend_by_best_rating()
        else:
            raise Exception('Selection method %s not yet handled.' % self.selection)
    
    def scrape_and_add_neighbours(self, title):
        # potentially storing a reason for the addition
        page = self.open_article(article_title=title)
        # page = wikipedia.WikipediaPage(title=title)
        neighbours = get_page_neighbours(page)
        added_new = False
        for new_title in neighbours:
            if new_title not in self.ratings:
                try:
                    new_page = self.open_article(article_title=new_title)
                    # new_page = wikipedia.WikipediaPage(title=new_title)
                    new_features = self.update_feature_extraction(new_page)
                    # new_features = self.extract_features(new_page)
                    self.ratings[new_title] = {'features': new_features, 'rated': 0}
                    added_new = True
                    print('Added article: ', new_title)
                except Exception as e:
                    print('Exception raised for article: ', new_title)
                    print(e)
                time.sleep(0.1)
        return added_new
    
    def present_summary(self, page):
        # just prints to console
        print(get_page_summary(page))
    
    def present_full_article(self, page):
        # just prints to console
        print(get_page_content(page))
    
    def predict_rating(self, features):
        # using cosine similarity:
        from numpy import array
        from numpy import exp as e_  # avoid potential name conflicts
        rated_items = [
            # TODO: update features when adding new pages, and normalize
            #       feature vectors
            (self._get_vector_from_features(v['features']), v['rated'])
            for v in self.ratings.values()
            if v['rated'] != 0
            ]
        rated_features = array([x[0] for x in rated_items])
        rated_scores = array([x[1] for x in rated_items])
        feature_vec = array(
            self._get_vector_from_features(features)
            ).reshape(-1, 1)
        similarities = rated_features @ feature_vec
        # this softmax means that every second-article-candidate
        # will have the same predicted rating as the first article
        # as the weights won't have any effect
        similarities = e_(similarities)
        ### DEBUG
        if similarities.sum() <= 0:
            print('Similarities sum to a non-positive')
            print(similarities)
        ###
        similarities = similarities / similarities.sum()
        rating = similarities.ravel().dot(rated_scores.ravel())
        return rating
    
    def _get_vector_from_features(self, features):
        # features: [(idx, score)]
        # return vec
        from numpy import zeros
        from numpy import log as l_  # avoid potential name conflicts
        feature_vec = zeros(len(self.words))
        for idx, count in features:
            tfidf = count / l_(self.words[self.word_index[idx]][2] + self.alpha)
            feature_vec[idx] = tfidf
        feature_vec = feature_vec / feature_vec.dot(feature_vec)
        return feature_vec
    
    def update_feature_extraction(self, page):
        # returns features of page
        words = word_tokenize(get_page_content(page))
        words = [
            self.lemmatizer.lemmatize(w.lower(), 'n') for w in words
            if w not in punctuation
            ]
        words = [self.lemmatizer.lemmatize(w, 'v') for w in words]
        if not words:
            raise Exception('Page %s has no extracted words' % get_page_title(page))
        
        word_counts = [(w, words.count(w)) for w in set(words)]
        feature_vector = []
        for w, count in word_counts:
            if w not in self.words:
                feature_vector.append([len(self.words), count])
                self.words[w] = [len(self.words), count, 1]
                self.word_index[len(self.words)-1] = w
            else:
                lookup = self.words[w]
                feature_vector.append(
                    [lookup[0], (lookup[1] + count)]
                    )
                self.words[w] = [
                    lookup[0],
                    lookup[1] + count,
                    lookup[2] + 1
                    ]
        return feature_vector


if __name__ == '__main__':
    print('starting...\nplease consider donating to wikipedia')
    rater = ratings_state()
    rater.run()
