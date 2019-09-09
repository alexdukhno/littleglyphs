import numpy as np
import scipy
import littleglyphs as lilg

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
        [lilg.FeatureBezierCurve(),lilg.FeatureLineSegment()],
        category=2
    )
    glyph.features[0].set_values([0.3,0.4,0.8,0.2,0.3,0.8,2])
    glyph.features[1].set_values([0.2,0.8,0.7,0.8])
    glyphs.append(glyph)    

    glyph = lilg.Glyph(
        [lilg.FeatureBezierCurve(),lilg.FeatureBezierCurve(),],
        category=3
    )
    glyph.features[0].set_values([0.3,0.2,0.8,0.35,0.3,0.5,1.5])
    glyph.features[1].set_values([0.3,0.5,0.8,0.65,0.3,0.8,1.5])
    glyphs.append(glyph)        
    
    glyph = lilg.Glyph(
        [lilg.FeatureLineSegment(),lilg.FeatureLineSegment(),lilg.FeatureLineSegment()],
        category=4
    )
    glyph.features[0].set_values([0.4,0.2,0.3,0.5])
    glyph.features[1].set_values([0.3,0.5,0.7,0.5])
    glyph.features[2].set_values([0.8,0.2,0.7,0.8])
    glyphs.append(glyph)        
    
    glyph = lilg.Glyph(
        [lilg.FeatureLineSegment(),lilg.FeatureLineSegment(),lilg.FeatureLineSegment()],
        category=4
    )
    glyph.features[0].set_values([0.6,0.2,0.3,0.6])
    glyph.features[1].set_values([0.3,0.6,0.7,0.6])
    glyph.features[2].set_values([0.6,0.2,0.6,0.8])
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
        [lilg.FeatureEllipse(),lilg.FeatureEllipse(),],
        category=8
    )
    glyph.features[0].set_values([0.3,0.45,0.7,0.8])
    glyph.features[1].set_values([0.3,0.45,0.7,0.2])
    glyphs.append(glyph)     


    glyph = lilg.Glyph(
        [lilg.FeatureEllipse(),lilg.FeatureBezierCurve(),],
        category=9
    )
    glyph.features[0].set_values([0.3,0.55,0.65,0.2])
    glyph.features[1].set_values([0.65,0.45,0.7,0.75,0.3,0.75,1.2])
    glyphs.append(glyph)    
    
    #[lilg.FeatureBezierCurve() for count in range(0,N_bezier_features)]+
    #[lilg.FeatureLineSegment() for count in range(0,N_line_features)]+
    # [lilg.FeatureEllipse() for count in range(0,N_ellipse_features)]

    glyph_alphabet = lilg.GlyphList(glyphs)
    return glyph_alphabet