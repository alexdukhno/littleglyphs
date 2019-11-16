import numpy as np
import scipy
import littleglyphs as lilg
from . import generation as lilggen



def MNISTlike_glyph_alphabet():
    glyphs = []

    glyph = lilg.Glyph(
        [lilg.FeatureEllipse()],
        category=0
    )
    glyph.features[0].set_values([0.25,0.2,0.75,0.8])
    glyphs.append(glyph)
    
    
    glyph = lilg.Glyph(
        [lilg.FeatureLineSegment()],
        category=1
    )
    glyph.features[0].set_values([0.5,0.2,0.5,0.8])
    glyphs.append(glyph)

    
    glyph = lilg.Glyph(
        [lilg.FeatureLineSegment(),lilg.FeatureLineSegment()],
        category=1
    )
    glyph.features[0].set_values([0.5,0.2,0.5,0.8])
    glyph.features[1].set_values([0.5,0.2,0.3,0.4])
    glyphs.append(glyph)

    
    glyph = lilg.Glyph(
        [lilg.FeatureMultiPointBezierCurve(3)],
        category=2
    )
    glyph.features[0].set_values([0.3,0.4,
                                  0.8,0.2,
                                  0.3,0.8,
                                  2,
                                  0.5,0.6,
                                  0.7,0.8,
                                  0.2
                                 ])
    glyphs.append(glyph)    

    
    glyph = lilg.Glyph(
        [lilg.FeatureMultiPointBezierCurve(4)],
        category=2
    )
    glyph.features[0].set_values([0.3,0.4,
                                  0.8,0.2,
                                  0.3,0.8,
                                  2,
                                  0.2,0.5,
                                  0.2,0.7,
                                  0.5,
                                  0.5,0.6,
                                  0.7,0.8,
                                  0.2
                                 ])
    glyphs.append(glyph)    
        
    
    glyph = lilg.Glyph(
        [lilg.FeatureMultiPointBezierCurve(3),],
        category=3
    )
    glyph.features[0].set_values([0.3,0.2,
                                  0.8,0.35,
                                  0.3,0.5,
                                  1.5,
                                  0.8,0.65,
                                  0.3,0.8,
                                  1.5
                                 ])
    glyphs.append(glyph)            
    
    
    glyph = lilg.Glyph(
        [lilg.FeatureMultiPointLineSegment(3),lilg.FeatureLineSegment()],
        category=4
    )
    glyph.features[0].set_values([0.4,0.2,
                                  0.3,0.5,
                                  0.7,0.5
                                 ]
                                )
    glyph.features[1].set_values([0.8,0.2,0.7,0.8])
    glyphs.append(glyph)        
    
    glyph = lilg.Glyph(
        [lilg.FeatureMultiPointLineSegment(4)],
        category=4
    )
    glyph.features[0].set_values([0.7,0.6,
                                  0.3,0.6,
                                  0.6,0.2,
                                  0.6,0.8
                                 ]
                                )
    glyphs.append(glyph) 

    glyph = lilg.Glyph(
        [lilg.FeatureLineSegment(),lilg.FeatureLineSegment(),lilg.FeatureBezierCurve(),],
        category=5
    )
    glyph.features[0].set_values([0.3,0.2,0.7,0.2])
    glyph.features[1].set_values([0.3,0.2,0.3,0.45])
    glyph.features[2].set_values([0.3,0.45,0.8,0.65,0.3,0.8,1.5])
    glyphs.append(glyph) 
    
    glyph = lilg.Glyph(
        [lilg.FeatureEllipse(),lilg.FeatureBezierCurve(),],
        category=6
    )
    glyph.features[0].set_values([0.3,0.45,0.7,0.8])
    glyph.features[1].set_values([0.3,0.5,0.4,0.25,0.7,0.2,1.5])
    glyphs.append(glyph) 
    
    glyph = lilg.Glyph(
        [lilg.FeatureMultiPointBezierCurve(4),],
        category=6
    )
    glyph.features[0].set_values([0.7,0.2,
                                  0.4,0.25,
                                  0.3,0.5,
                                  1.5,
                                  0.2,0.7,
                                  0.6,0.7,
                                  1.0,
                                  0.6,0.4,
                                  0.3,0.5,
                                  1.0
                                 ])
    glyphs.append(glyph)     
    
    glyph = lilg.Glyph(
        [lilg.FeatureLineSegment(),lilg.FeatureLineSegment()],
        category=7
    )
    glyph.features[0].set_values([0.3,0.25,0.7,0.25])
    glyph.features[1].set_values([0.7,0.25,0.35,0.75])
    glyphs.append(glyph) 

    glyph = lilg.Glyph(
        [lilg.FeatureLineSegment(),lilg.FeatureLineSegment(),lilg.FeatureLineSegment()],
        category=7
    )
    glyph.features[0].set_values([0.3,0.25,0.7,0.25])
    glyph.features[1].set_values([0.7,0.25,0.35,0.75])
    glyph.features[2].set_values([0.35,0.45,0.75,0.5])
    glyphs.append(glyph) 

    glyph = lilg.Glyph(
        [lilg.FeatureMultiPointBezierCurve(5),],
        category=8
    )
    glyph.features[0].set_values([0.5,0.25,
                                  0.2,0.35,
                                  0.5,0.5,
                                  1.2,
                                  0.8,0.65,
                                  0.5,0.75,
                                  1.2,
                                  0.2,0.65,
                                  0.5,0.5,
                                  1.2,
                                  0.8,0.35,
                                  0.5,0.25,
                                  1.2
                                 ])
    glyphs.append(glyph)    


    glyph = lilg.Glyph(
        [lilg.FeatureEllipse(),lilg.FeatureBezierCurve(),],
        category=9
    )
    glyph.features[0].set_values([0.3,0.55,0.65,0.2])
    glyph.features[1].set_values([0.65,0.45,0.7,0.75,0.3,0.75,1.2])
    glyphs.append(glyph)    
    
    
    glyph = lilg.Glyph(
        [lilg.FeatureMultiPointBezierCurve(4),],
        category=9
    )
    glyph.features[0].set_values([0.65,0.45,
                                  0.2,0.6,
                                  0.35,0.25,
                                  1.5,
                                  0.7,0.2,
                                  0.65,0.45,
                                  1.5,
                                  0.7,0.75,
                                  0.3,0.75,
                                  1.2
                                 ])
    glyphs.append(glyph)     
        
    
    
    
    glyph_alphabet = lilg.GlyphList(glyphs)
    return glyph_alphabet




def basic_random_glyph_generator(blocky=False):
    glyph_gens = []
    glyph_gens.append(lilggen.RandomGlyphGenerator(blocky=blocky))
    glyph_gens[0].add_feature(lilg.FeatureBezierCurve, feature_count=2)
    glyph_gens.append(lilggen.RandomGlyphGenerator(blocky=blocky))
    glyph_gens[1].add_feature(lilg.FeatureBezierCurve, feature_count=3)
    glyph_gens.append(lilggen.RandomGlyphGenerator(blocky=blocky))
    glyph_gens[2].add_feature(lilg.FeatureMultiPointBezierCurve, feature_count=1, N_points=3)
    
    alphabet_generator = lilggen.RandomAlphabetGenerator(glyph_gens,weights=[1,0.5,1])
    return alphabet_generator