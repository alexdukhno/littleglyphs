import copy
import numpy as np
import scipy

import littleglyphs as lilg



class RandomGlyphGenerator():
    '''
    RandomGlyphGenerator gets fed a list of feature types and their counts.
    From these, it produces a randomized glyph each time its generate_random_glyph method is called.
    
    Example usage:
        glyph_gen = GlyphGenerator()
        glyph_gen.add_feature(lilg.FeatureEllipse, feature_count=1)
        glyph_gen.add_feature(lilg.FeatureMultiPointBezierCurve, feature_count=1, N_points=4)
        g = glyph_gen.generate_random_glyph(category=0)
    '''    
    
    feature_list = None
    blocky = False
    
    def __init__(self, blocky=False): 
        self.feature_list = []
        self.blocky = blocky
    
    def add_feature(self, feature_type, feature_count=1, **kwargs):
        self.feature_list += [feature_type(blocky=self.blocky,**kwargs) for count in range(0,feature_count)]
            
    def generate_random_glyph(self, category):
        glyph = lilg.Glyph(self.feature_list)
        glyph.set_category(category)
        glyph.randomize_all_features()
        return copy.deepcopy(glyph)
    

    

class RandomAlphabetGenerator():
    '''
    RandomAlphabetGenerator gets fed a list of glyph generators and (optionally) their weights.
    From these, it produces a random glyph alphabet, choosing one of its random generators with according weight.
    
    '''
    
    glyph_generator_list = None
    weights = None
    N_glyph_generators = None
    
    def __init__(self, glyph_generator_list, weights=None): 
        if glyph_generator_list is None:
            raise('RandomAlphabetGenerator needs a list of at least one glyph generator to work')            
        if not isinstance(glyph_generator_list, list):
            self.glyph_generator_list = [glyph_generator_list]
        else:        
            self.glyph_generator_list = glyph_generator_list
            
        self.glyph_generator_list = copy.deepcopy(self.glyph_generator_list)
        self.N_glyph_generators = len(self.glyph_generator_list)
        
        if (weights is None) or (weights == []):
            self.weights = np.ones(self.N_glyph_generators) / self.N_glyph_generators
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / np.sum(self.weights)
        
    def generate_random_glyph_alphabet(self, categories):
        glyphs = []
        for category in categories:
            chosen_generator = self.glyph_generator_list[np.random.choice(self.N_glyph_generators,p=self.weights)]
            glyph = chosen_generator.generate_random_glyph(category)
            glyphs.append(glyph)
        glyph_alphabet = lilg.GlyphList(glyphs)
        return copy.deepcopy(glyph_alphabet)